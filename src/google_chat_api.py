"""
Google API Client - Handles all communication with Google's Gemini API.
This module is used by both OpenAI compatibility layer and native Gemini endpoints.
"""
import asyncio
import gc
import json
from typing import Optional
import re
import time
from datetime import datetime, timezone, timedelta

from fastapi import Response
from fastapi.responses import StreamingResponse

from config import (
    get_code_assist_endpoint,
    DEFAULT_SAFETY_SETTINGS,
    get_base_model_name,
    get_thinking_budget,
    should_include_thoughts,
    is_search_model,
    get_auto_ban_enabled,
    get_auto_ban_error_codes,
    get_retry_429_max_retries,
    get_retry_429_enabled,
    get_retry_429_interval,
    PUBLIC_API_MODELS
)
from .httpx_client import http_client, create_streaming_client_with_kwargs
from log import log
from .credential_manager import CredentialManager
from .usage_stats import record_successful_call
from .utils import get_user_agent


def _parse_duration(duration_str: str) -> Optional[timedelta]:
    """
    Parses a duration string (e.g., '10h17m47.175s', '600s') into a timedelta object.
    """
    if not duration_str or not isinstance(duration_str, str):
        return None

    # 移除 's' 后缀并检查是否为纯数字
    if duration_str.endswith('s') and duration_str[:-1].replace('.', '', 1).isdigit():
        try:
            return timedelta(seconds=float(duration_str[:-1]))
        except ValueError:
            return None

    # 用于解析 1h2m3.456s 格式的正则表达式
    pattern = re.compile(r'(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+(?:\.\d+)?)s)?')
    match = pattern.match(duration_str)

    if not match:
        return None

    parts = match.groupdict()
    time_params = {}
    for (name, param) in parts.items():
        if param:
            time_params[name] = float(param)

    if not time_params:
        return None

    return timedelta(**time_params)

async def _calculate_reset_timestamp(response_content: str) -> Optional[float]:
    """
    Parses 429 error response to find quotaResetTimeStamp or quotaResetDelay
    and calculates the final Unix timestamp for unbanning.
    """
    if not response_content:
        return None

    try:
        error_data = json.loads(response_content)
        details = error_data.get('error', {}).get('details', [{}])[0]
        metadata = details.get('metadata', {})

        # 优先使用绝对时间戳
        if 'quotaResetTimeStamp' in metadata:
            timestamp_str = metadata['quotaResetTimeStamp']
            try:
                # 解析ISO 8601格式的时间戳 (e.g., '2024-05-15T12:00:00.123456Z')
                # Python's fromisoformat can handle 'Z' correctly since 3.11
                if timestamp_str.endswith('Z'):
                    dt = datetime.fromisoformat(timestamp_str[:-1] + '+00:00')
                else:
                    dt = datetime.fromisoformat(timestamp_str)

                # 如果没有时区信息，则假定为UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

                return dt.timestamp()
            except ValueError as e:
                log.warning(f"Failed to parse quotaResetTimeStamp '{timestamp_str}': {e}")
                return None

        # 其次使用相对时间间隔
        elif 'quotaResetDelay' in metadata:
            delay_str = metadata['quotaResetDelay']
            duration = _parse_duration(delay_str)
            if duration:
                return time.time() + duration.total_seconds()
            else:
                log.warning(f"Failed to parse quotaResetDelay '{delay_str}'")
                return None

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        log.debug(f"Could not extract reset timestamp from response: {e}")
        return None

    return None

def _create_error_response(message: str, status_code: int = 500) -> Response:
    """Create standardized error response."""
    return Response(
        content=json.dumps({
            "error": {
                "message": message,
                "type": "api_error",
                "code": status_code
            }
        }),
        status_code=status_code,
        media_type="application/json"
    )

async def _handle_api_error(credential_manager: CredentialManager, status_code: int, response_content: str = ""):
    """Handle API errors by rotating credentials when needed. Error recording should be done before calling this function."""
    # The primary logging is now done at the point of failure detection with more context (like credential index).
    # This function is kept for potential future logic (like global error handling) but its logging part is removed to avoid duplicates.
    pass

async def _prepare_request_headers_and_payload(payload: dict, credential_data: dict, use_public_api: bool, target_url: str):
    """Prepare request headers and final payload from credential data."""
    token = credential_data.get('token') or credential_data.get('access_token', '')
    if not token:
        raise Exception("凭证中没有找到有效的访问令牌（token或access_token字段）")


    source_request=payload.get("request", {})
    if use_public_api:
         if "generationConfig" in source_request:
            imageConfig = source_request["generationConfig"].get('imageConfig')
            source_request["generationConfig"] = {'imageConfig': imageConfig} if imageConfig else {}
    
    # 内部API使用Bearer Token和项目ID
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    project_id = credential_data.get("project_id", "")
    if not project_id:
        raise Exception("项目ID不存在于凭证数据中")
    final_payload = {
        "model": payload.get("model"),
        "project": project_id,
        "request": source_request
    }
    
    return headers, final_payload, target_url

async def send_gemini_request(payload: dict, is_streaming: bool = False, credential_manager: CredentialManager = None) -> Response:
    """
    Send a request to Google's Gemini API.
    This version removes the internal retry loop, relying on the credential manager for 429 handling.
    """
    # 动态确定API端点和payload格式
    model_name = payload.get("model", "")
    base_model_name = get_base_model_name(model_name)
    use_public_api = base_model_name in PUBLIC_API_MODELS
    action = "streamGenerateContent" if is_streaming else "generateContent"
    target_url = f"{await get_code_assist_endpoint()}/v1internal:{action}"
    if is_streaming:
        target_url += "?alt=sse"

    # 确保有credential_manager
    if not credential_manager:
        return _create_error_response("Credential manager not provided", 500)

    # 获取当前凭证
    try:
        credential_result = await credential_manager.get_valid_credential()
        if not credential_result:
            return _create_error_response("No valid credentials available", 500)

        current_file, credential_data, current_index = credential_result
        headers, final_payload, target_url = await _prepare_request_headers_and_payload(payload, credential_data, use_public_api, target_url)
    except Exception as e:
        return _create_error_response(str(e), 500)

    # 预序列化payload
    final_post_data = json.dumps(final_payload)

    try:
        if is_streaming:
            # 流式请求处理
            client = await create_streaming_client_with_kwargs()
            try:
                stream_ctx = client.stream("POST", target_url, content=final_post_data, headers=headers)
                resp = await stream_ctx.__aenter__()

                # 不再内部处理429，直接传递给上层
                if resp.status_code != 200:
                    response_content = ""
                    try:
                        content_bytes = await resp.aread()
                        if isinstance(content_bytes, bytes):
                            response_content = content_bytes.decode('utf-8', errors='ignore')
                    except Exception as e:
                        log.debug(f"[STREAMING] Failed to read error response content: {e}")

                    cred_identifier = f"索引 {current_index}" if current_index != -1 else current_file

                    if response_content:
                        log.error(f"凭证 {cred_identifier} 的Google API返回状态 {resp.status_code} (STREAMING). 响应详情: {response_content[:500]}")
                    else:
                        log.error(f"凭证 {cred_identifier} 的Google API返回状态 {resp.status_code} (STREAMING)")

                    if credential_manager and current_file:
                        reset_timestamp = await _calculate_reset_timestamp(response_content)
                        await credential_manager.record_api_call_result(
                            credential_name=current_file,
                            success=False,
                            error_code=resp.status_code,
                            temp_disabled_until=reset_timestamp
                        )

                    await _handle_api_error(credential_manager, resp.status_code, response_content)

                    # 清理资源并返回错误
                    try:
                        await stream_ctx.__aexit__(None, None, None)
                    except: pass
                    await client.aclose()

                    async def error_stream():
                        error_response = {"error": {"message": f"API error: {resp.status_code}", "type": "api_error", "code": resp.status_code}}
                        yield f"data: {json.dumps(error_response)}\n\n"
                    return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=resp.status_code)
                else:
                    # 成功响应，传递所有资源给流式处理函数管理
                    return _handle_streaming_response_managed(resp, stream_ctx, client, credential_manager, payload.get("model", ""), current_file, current_index)

            except Exception as e:
                try:
                    await client.aclose()
                except: pass
                raise e
        else:
            # 非流式请求处理
            async with http_client.get_client(timeout=None) as client:
                resp = await client.post(
                    target_url, content=final_post_data, headers=headers
                )
                # 直接将响应（无论成功失败）传递给处理函数
                return await _handle_non_streaming_response(resp, credential_manager, payload.get("model", ""), current_file, current_index)

    except Exception as e:
        log.error(f"Request to Google API failed: {str(e)}")
        return _create_error_response(f"Request failed: {str(e)}")


def _handle_streaming_response_managed(resp, stream_ctx, client, credential_manager: CredentialManager = None, model_name: str = "", current_file: str = None, current_index: int = -1) -> StreamingResponse:
    """Handle streaming response with complete resource lifecycle management."""
    
    # 检查HTTP错误
    if resp.status_code != 200:
        # 立即清理资源并返回错误
        async def cleanup_and_error():
            try:
                await stream_ctx.__aexit__(None, None, None)
            except:
                pass
            try:
                await client.aclose()
            except:
                pass
            
            # 获取响应内容用于详细错误显示
            response_content = ""
            try:
                content_bytes = await resp.aread()
                if isinstance(content_bytes, bytes):
                    response_content = content_bytes.decode('utf-8', errors='ignore')
            except Exception as e:
                log.debug(f"[STREAMING] Failed to read response content for error analysis: {e}")
                response_content = ""
            
            cred_identifier = f"索引 {current_index}" if current_index != -1 else current_file
            # 显示详细错误信息
            if resp.status_code == 429:
                if response_content:
                    log.error(f"凭证 {cred_identifier} 的Google API返回状态 429 (STREAMING). 响应详情: {response_content[:500]}")
                else:
                    log.error(f"凭证 {cred_identifier} 的Google API返回状态 429 (STREAMING)")
            else:
                if response_content:
                    log.error(f"凭证 {cred_identifier} 的Google API返回状态 {resp.status_code} (STREAMING). 响应详情: {response_content[:500]}")
                else:
                    log.error(f"凭证 {cred_identifier} 的Google API返回状态 {resp.status_code} (STREAMING)")
            
            # 记录API调用错误
            if credential_manager and current_file:
                reset_timestamp = await _calculate_reset_timestamp(response_content)
                await credential_manager.record_api_call_result(
                    credential_name=current_file,
                    success=False,
                    error_code=resp.status_code,
                    temp_disabled_until=reset_timestamp
                )
            
            await _handle_api_error(credential_manager, resp.status_code, response_content)
            
            error_response = {
                "error": {
                    "message": f"API error: {resp.status_code}",
                    "type": "api_error",
                    "code": resp.status_code
                }
            }
            yield f'data: {json.dumps(error_response)}\n\n'.encode('utf-8')
        
        return StreamingResponse(
            cleanup_and_error(),
            media_type="text/event-stream",
            status_code=resp.status_code
        )
    
    # 正常流式响应处理，确保资源在流结束时被清理
    async def managed_stream_generator():
        success_recorded = False
        managed_stream_generator._chunk_count = 0  # 初始化chunk计数器
        try:
            async for chunk in resp.aiter_lines():
                if not chunk or not chunk.startswith('data: '):
                    continue
                    
                # 记录第一次成功响应
                if not success_recorded:
                    if current_file and credential_manager:
                        await credential_manager.record_api_call_result(current_file, True)
                        # 记录到使用统计
                        try:
                            await record_successful_call(current_file, model_name)
                        except Exception as e:
                            log.debug(f"Failed to record usage statistics: {e}")
                    success_recorded = True
                
                payload = chunk[len('data: '):]
                try:
                    obj = json.loads(payload)
                    if "response" in obj:
                        data = obj["response"]
                        yield f"data: {json.dumps(data, separators=(',',':'))}\n\n".encode()
                        await asyncio.sleep(0)  # 让其他协程有机会运行
                        
                        # 定期释放内存（每100个chunk）
                        if hasattr(managed_stream_generator, '_chunk_count'):
                            managed_stream_generator._chunk_count += 1
                            if managed_stream_generator._chunk_count % 100 == 0:
                                gc.collect()
                    else:
                        yield f"data: {json.dumps(obj, separators=(',',':'))}\n\n".encode()
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            log.error(f"Streaming error: {e}")
            err = {"error": {"message": str(e), "type": "api_error", "code": 500}}
            yield f"data: {json.dumps(err)}\n\n".encode()
        finally:
            # 确保清理所有资源
            try:
                await stream_ctx.__aexit__(None, None, None)
            except Exception as e:
                log.debug(f"Error closing stream context: {e}")
            try:
                await client.aclose()
            except Exception as e:
                log.debug(f"Error closing client: {e}")

    return StreamingResponse(
        managed_stream_generator(),
        media_type="text/event-stream"
    )

async def _handle_non_streaming_response(resp, credential_manager: CredentialManager = None, model_name: str = "", current_file: str = None, current_index: int = -1) -> Response:
    """Handle non-streaming response from Google API."""
    if resp.status_code == 200:
        try:
            # 记录成功响应
            if current_file and credential_manager:
                await credential_manager.record_api_call_result(current_file, True)
                # 记录到使用统计
                try:
                    await record_successful_call(current_file, model_name)
                except Exception as e:
                    log.debug(f"Failed to record usage statistics: {e}")
            
            raw = await resp.aread()
            google_api_response = raw.decode('utf-8')
            if google_api_response.startswith('data: '):
                google_api_response = google_api_response[len('data: '):]
            google_api_response = json.loads(google_api_response)
            log.debug(f"Google API原始响应: {json.dumps(google_api_response, ensure_ascii=False)[:500]}...")
            standard_gemini_response = google_api_response.get("response")
            log.debug(f"提取的response字段: {json.dumps(standard_gemini_response, ensure_ascii=False)[:500]}...")
            return Response(
                content=json.dumps(standard_gemini_response),
                status_code=200,
                media_type="application/json; charset=utf-8"
            )
        except Exception as e:
            log.error(f"Failed to parse Google API response: {str(e)}")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("Content-Type")
            )
    else:
        # 获取响应内容用于详细错误显示
        response_content = ""
        try:
            if hasattr(resp, 'content'):
                content = resp.content
                if isinstance(content, bytes):
                    response_content = content.decode('utf-8', errors='ignore')
            else:
                content_bytes = await resp.aread()
                if isinstance(content_bytes, bytes):
                    response_content = content_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            log.debug(f"[NON-STREAMING] Failed to read response content for error analysis: {e}")
            response_content = ""
        
        cred_identifier = f"索引 {current_index}" if current_index != -1 else current_file
        # 显示详细错误信息
        if resp.status_code == 429:
            if response_content:
                log.error(f"凭证 {cred_identifier} 的Google API返回状态 429 (NON-STREAMING). 响应详情: {response_content[:500]}")
            else:
                log.error(f"凭证 {cred_identifier} 的Google API返回状态 429 (NON-STREAMING)")
        else:
            if response_content:
                log.error(f"凭证 {cred_identifier} 的Google API返回状态 {resp.status_code} (NON-STREAMING). 响应详情: {response_content[:500]}")
            else:
                log.error(f"凭证 {cred_identifier} 的Google API返回状态 {resp.status_code} (NON-STREAMING)")
        
        # 记录API调用错误
        if credential_manager and current_file:
            reset_timestamp = await _calculate_reset_timestamp(response_content)
            await credential_manager.record_api_call_result(
                credential_name=current_file,
                success=False,
                error_code=resp.status_code,
                temp_disabled_until=reset_timestamp
            )
        
        await _handle_api_error(credential_manager, resp.status_code, response_content)
        
        return _create_error_response(f"API error: {resp.status_code}", resp.status_code)

def build_gemini_payload_from_native(native_request: dict, model_from_path: str) -> dict:
    """
    Build a Gemini API payload from a native Gemini request with full pass-through support.
    """
    # 创建请求副本以避免修改原始数据
    request_data = native_request.copy()
    
    # 应用默认安全设置（如果未指定）
    if "safetySettings" not in request_data:
        request_data["safetySettings"] = DEFAULT_SAFETY_SETTINGS
    
    # 确保generationConfig存在
    if "generationConfig" not in request_data:
        request_data["generationConfig"] = {}
    
    generation_config = request_data["generationConfig"]
    
    # 配置thinking（如果未指定thinkingConfig）
    if "thinkingConfig" not in generation_config:
        generation_config["thinkingConfig"] = {}
    
    thinking_config = generation_config["thinkingConfig"]
    
    # 只有在未明确设置时才应用默认thinking配置
    if "includeThoughts" not in thinking_config:
        thinking_config["includeThoughts"] = should_include_thoughts(model_from_path)
    if "thinkingBudget" not in thinking_config:
        thinking_config["thinkingBudget"] = get_thinking_budget(model_from_path)
    
    # 为搜索模型添加Google Search工具（如果未指定且没有functionDeclarations）
    if is_search_model(model_from_path):
        if "tools" not in request_data:
            request_data["tools"] = []
        # 检查是否已有functionDeclarations或googleSearch工具
        has_function_declarations = any(tool.get("functionDeclarations") for tool in request_data["tools"])
        has_google_search = any(tool.get("googleSearch") for tool in request_data["tools"])
        
        # 只有在没有任何工具时才添加googleSearch，或者只有googleSearch工具时可以添加更多googleSearch
        if not has_function_declarations and not has_google_search:
            request_data["tools"].append({"googleSearch": {}})
    
    # 透传所有其他Gemini原生字段:
    # - contents (必需)
    # - systemInstruction (可选)
    # - generationConfig (已处理)
    # - safetySettings (已处理)  
    # - tools (已处理)
    # - toolConfig (透传)
    # - cachedContent (透传)
    # - 以及任何其他未知字段都会被透传
    
    return {
        "model": get_base_model_name(model_from_path),
        "request": request_data
    }
