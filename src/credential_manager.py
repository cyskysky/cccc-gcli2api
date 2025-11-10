"""
凭证管理器 - 完全基于统一存储中间层
"""
import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager

from config import get_calls_per_rotation, is_mongodb_mode
from log import log
from .storage_adapter import get_storage_adapter
from .google_oauth_api import fetch_user_email_from_file, Credentials
from .task_manager import task_manager


class CredentialManager:
    """
    统一凭证管理器
    所有存储操作通过storage_adapter进行
    """
    
    def __init__(self):
        # 核心状态
        self._initialized = False
        self._storage_adapter = None
        
        # 凭证轮换相关
        self._credential_files: List[str] = []  # 存储凭证文件名列表
        self._current_credential_index = 0
        self._last_scan_time = 0
        
        # 当前使用的凭证信息
        self._current_credential_file: Optional[str] = None
        self._current_credential_data: Optional[Dict[str, Any]] = None
        self._current_credential_state: Dict[str, Any] = {}
        
        # 并发控制
        self._state_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        
        # 工作线程控制
        self._shutdown_event = asyncio.Event()
        self._write_worker_running = False
        self._write_worker_task = None
        
        # 原子操作计数器
        self._atomic_counter = 0
        self._atomic_lock = asyncio.Lock()
        
        # Onboarding state
        self._onboarding_complete = False
        self._onboarding_checked = False
    
    async def initialize(self):
        """初始化凭证管理器"""
        async with self._state_lock:
            if self._initialized:
                return
            
            # 初始化统一存储适配器
            self._storage_adapter = await get_storage_adapter()
            
            # 启动后台工作线程
            await self._start_background_workers()
            
            # 发现并加载凭证
            await self._discover_credentials()
            
            self._initialized = True
            storage_type = "MongoDB" if await is_mongodb_mode() else "File"
            log.debug(f"Credential manager initialized with {storage_type} storage backend")
    
    async def close(self):
        """清理资源"""
        log.debug("Closing credential manager...")
        
        # 设置关闭标志
        self._shutdown_event.set()
        
        # 等待后台任务结束
        if self._write_worker_task:
            try:
                await asyncio.wait_for(self._write_worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                log.warning("Write worker task did not finish within timeout")
                if not self._write_worker_task.done():
                    self._write_worker_task.cancel()
            except asyncio.CancelledError:
                # 任务被取消是正常的关闭流程
                log.debug("Background worker task was cancelled during shutdown")

        self._initialized = False
        log.debug("Credential manager closed")
    
    async def _start_background_workers(self):
        """启动后台工作线程"""
        if not self._write_worker_running:
            self._write_worker_running = True
            self._write_worker_task = task_manager.create_task(
                self._background_worker(), 
                name="credential_background_worker"
            )
    
    async def _background_worker(self):
        """后台工作线程，处理定期任务"""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # 每60秒检查一次凭证更新
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=60.0)
                    if self._shutdown_event.is_set():
                        break

                    # 重新发现凭证（热更新）
                    await self._discover_credentials()

                except asyncio.TimeoutError:
                    # 超时是正常的，继续下一轮
                    continue
                except asyncio.CancelledError:
                    # 任务被取消，正常退出
                    log.debug("Background worker cancelled, exiting gracefully")
                    break
                except Exception as e:
                    log.error(f"Background worker error: {e}")
                    await asyncio.sleep(5)  # 错误后等待5秒再继续
        except asyncio.CancelledError:
            # 外层捕获取消，确保干净退出
            log.debug("Background worker received cancellation")
        finally:
            log.debug("Background worker exited")
            self._write_worker_running = False
    
    async def _discover_credentials(self):
        """发现和加载所有可用凭证，并处理两种禁用恢复逻辑"""
        try:
            # 1. 从存储中获取所有凭证文件和状态
            all_credentials_from_storage = await self._storage_adapter.list_credentials()
            all_states = await self._storage_adapter.get_all_credential_states()

            # 2. 处理禁用恢复逻辑
            for cred_name in all_credentials_from_storage:
                state = all_states.get(cred_name, {})
                if not state.get("disabled", False):
                    continue

                # 统一恢复逻辑：只检查temp_disabled_until
                temp_disabled_until = state.get("temp_disabled_until")
                if temp_disabled_until and time.time() > temp_disabled_until:
                    china_tz = timezone(timedelta(hours=8))
                    human_readable_time = datetime.fromtimestamp(temp_disabled_until, tz=china_tz).strftime('%Y-%m-%d %H:%M:%S')
                    log.info(f"凭证 {cred_name} 的临时禁用已到期（解禁时间: {human_readable_time} 北京时间），恢复。")
                    await self.update_credential_state(cred_name, {
                        "disabled": False,
                        "temp_disabled_until": None,
                        "error_codes": [],
                    })

            # 3. 更新内存中的凭证文件列表 (只增不减)
            is_initial_load = self._last_scan_time == 0
            current_files_set = set(self._credential_files)
            all_storage_files_set = set(all_credentials_from_storage)

            if is_initial_load:
                self._credential_files = sorted(list(all_storage_files_set))
                log.debug(f"首次加载，发现 {len(self._credential_files)} 个凭证。")
            else:
                new_files = all_storage_files_set - current_files_set
                if new_files:
                    self._credential_files.extend(sorted(list(new_files)))
                    log.info(f"发现并添加了新的凭证文件: {sorted(list(new_files))}")

            self._last_scan_time = time.time()
            log.debug(f"凭证发现完成，当前管理的凭证总数: {len(self._credential_files)}")

        except Exception as e:
            log.error(f"发现凭证时出错: {e}")

    async def _load_credential_by_name(self, credential_name: str) -> Optional[Dict[str, Any]]:
        """加载指定名称的凭证数据，包含token过期检测和自动刷新"""
        try:
            credential_data = await self._storage_adapter.get_credential(credential_name)
            if not credential_data:
                log.error(f"无法加载凭证数据: {credential_name}")
                return None

            if 'type' not in credential_data and all(key in credential_data for key in ['client_id', 'refresh_token']):
                credential_data['type'] = 'authorized_user'

            if "access_token" in credential_data and "token" not in credential_data:
                credential_data["token"] = credential_data["access_token"]
            if "scope" in credential_data and "scopes" not in credential_data:
                credential_data["scopes"] = credential_data["scope"].split()

            if await self._should_refresh_token(credential_data):
                log.debug(f"Token for {credential_name} needs refresh.")
                refreshed_data = await self._refresh_token(credential_data, credential_name)
                if not refreshed_data:
                    log.error(f"Token refresh failed for {credential_name}.")
                    return None
                return refreshed_data

            return credential_data
        except Exception as e:
            log.error(f"加载凭证 {credential_name} 时出错: {e}")
            return None

    async def get_valid_credential(self) -> Optional[Tuple[str, Dict[str, Any], int]]:
        """获取有效的凭证，同时返回其索引。"""
        async with self._operation_lock:
            if not self._credential_files:
                log.warning("凭证管理器中没有任何凭证，尝试重新发现。")
                await self._discover_credentials()
                if not self._credential_files:
                    log.error("重新发现后仍无凭证可用。")
                    return None

            all_states = await self._storage_adapter.get_all_credential_states()
            num_creds = len(self._credential_files)

            for i in range(num_creds):
                current_index = (self._current_credential_index + i) % num_creds
                credential_name = self._credential_files[current_index]

                state = all_states.get(credential_name, {})
                if state.get("disabled", False):
                    continue

                credential_data = await self._load_credential_by_name(credential_name)

                if credential_data:
                    # 成功获取，更新索引以便下次从下一个开始
                    self._current_credential_index = (current_index + 1) % num_creds

                    # 更新当前凭证的缓存信息
                    self._current_credential_file = credential_name
                    self._current_credential_data = credential_data
                    self._current_credential_state = state

                    log.info(f"选择凭证 {credential_name} (索引 {current_index})。")
                    return credential_name, credential_data, current_index
                else: # 加载失败意味着token刷新失败或文件有问题
                    log.warning(f"加载凭证 {credential_name} 失败，将禁用该凭证。")
                    await self.set_cred_disabled(credential_name, True)
                    # 更新本地状态副本以避免在此次调用中再次尝试
                    all_states[credential_name] = all_states.get(credential_name, {})
                    all_states[credential_name]['disabled'] = True

            log.error("所有凭证均尝试失败，无可用凭证。")
            return None

    async def force_rotate_credential(self):
        """
        强制轮换凭证。在新的模型中，这会简单地将索引推进一位，
        确保下一次调用会从列表中的下一个凭证开始尝试。
        """
        async with self._operation_lock:
            if not self._credential_files:
                return
            num_creds = len(self._credential_files)
            self._current_credential_index = (self._current_credential_index + 1) % num_creds
            log.info(f"接收到强制轮换请求。下次将从索引 {self._current_credential_index} 开始尝试。")
    
    async def update_credential_state(self, credential_name: str, state_updates: Dict[str, Any]):
        """更新凭证状态"""
        try:
            # 直接通过存储适配器更新状态
            success = await self._storage_adapter.update_credential_state(credential_name, state_updates)
            
            # 如果是当前使用的凭证，更新缓存
            if credential_name == self._current_credential_file:
                self._current_credential_state.update(state_updates)
            
            if success:
                log.debug(f"Updated credential state: {credential_name}")
            else:
                log.warning(f"Failed to update credential state: {credential_name}")
                
            return success
            
        except Exception as e:
            log.error(f"Error updating credential state {credential_name}: {e}")
            return False
    
    async def set_cred_disabled(self, credential_name: str, disabled: bool):
        """设置凭证的启用/禁用状态"""
        try:
            state_updates = {"disabled": disabled}
            success = await self.update_credential_state(credential_name, state_updates)
            
            if success:
                action = "disabled" if disabled else "enabled"
                log.info(f"Credential {action}: {credential_name}")
            
            return success
            
        except Exception as e:
            log.error(f"Error setting credential disabled state {credential_name}: {e}")
            return False
    
    async def get_creds_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有凭证的状态"""
        try:
            # 从存储适配器获取所有状态
            all_states = await self._storage_adapter.get_all_credential_states()
            return all_states
            
        except Exception as e:
            log.error(f"Error getting credential statuses: {e}")
            return {}
    
    async def get_or_fetch_user_email(self, credential_name: str) -> Optional[str]:
        """获取或获取用户邮箱地址"""
        try:
            # 首先检查缓存的状态
            state = await self._storage_adapter.get_credential_state(credential_name)
            cached_email = state.get("user_email")
            
            if cached_email:
                return cached_email
            
            # 如果没有缓存，从凭证数据获取
            credential_data = await self._storage_adapter.get_credential(credential_name)
            if not credential_data:
                return None
            
            # 尝试获取邮箱
            email = await fetch_user_email_from_file(credential_data)
            
            if email:
                # 缓存邮箱地址
                await self.update_credential_state(credential_name, {"user_email": email})
                return email
            
            return None
            
        except Exception as e:
            log.error(f"Error fetching user email for {credential_name}: {e}")
            return None
    
    async def record_api_call_result(self, credential_name: str, success: bool, error_code: Optional[int] = None, temp_disabled_until: Optional[float] = None):
        """
        记录API调用结果。
        429禁用逻辑统一为仅依赖API返回的精确重置时间。
        添加了并发锁以防止竞态条件。
        """
        async with self._operation_lock:
            try:
                current_state = await self._storage_adapter.get_credential_state(credential_name)
                if not current_state:
                    current_state = {}

                state_updates = {}

                if success:
                    # 成功后重置所有错误相关的状态
                    # 检查是否需要重置（之前是禁用状态或有错误码）
                    if current_state.get("disabled") or current_state.get("error_codes"):
                        state_updates["last_success"] = time.time()
                        state_updates["error_codes"] = []
                        state_updates["temp_disabled_until"] = None
                        state_updates["disabled"] = False # 确保解除禁用
                        log.info(f"凭证 {credential_name} 调用成功，重置所有错误状态。")

                elif error_code == 429:
                    # 为了UI显示，仍然记录429错误码
                    error_codes = current_state.get("error_codes", [])
                    if 429 not in error_codes:
                        error_codes.append(429)
                    state_updates["error_codes"] = error_codes

                    # 统一禁用逻辑：仅当API提供了精确重置时间时才禁用
                    if temp_disabled_until:
                        state_updates["temp_disabled_until"] = temp_disabled_until
                        state_updates["disabled"] = True
                        china_tz = timezone(timedelta(hours=8))
                        human_readable_time = datetime.fromtimestamp(temp_disabled_until, tz=china_tz).strftime('%Y-%m-%d %H:%M:%S')
                        log.info(f"凭证 {credential_name} 因429错误被动态禁用，直至: {human_readable_time} (北京时间)")
                    else:
                        # 如果API没有提供重置时间，则只记录错误，不禁用
                        log.warning(f"凭证 {credential_name} 收到429错误，但API未提供重置时间，本次不禁用。")

                elif error_code:
                    # 处理其他错误码
                    error_codes = current_state.get("error_codes", [])
                    if error_code not in error_codes:
                        error_codes.append(error_code)
                    state_updates["error_codes"] = error_codes[-10:] # 保留最近10个

                if state_updates:
                    await self.update_credential_state(credential_name, state_updates)

            except Exception as e:
                log.error(f"记录API调用结果时出错 {credential_name}: {e}")
    
    # 原子操作支持
    @asynccontextmanager
    async def _atomic_operation(self, operation_name: str):
        """原子操作上下文管理器"""
        async with self._atomic_lock:
            self._atomic_counter += 1
            operation_id = self._atomic_counter
            log.debug(f"开始原子操作[{operation_id}]: {operation_name}")
            
            try:
                yield operation_id
                log.debug(f"完成原子操作[{operation_id}]: {operation_name}")
            except Exception as e:
                log.error(f"原子操作[{operation_id}]失败: {operation_name} - {e}")
                raise
    
    async def _should_refresh_token(self, credential_data: Dict[str, Any]) -> bool:
        """检查token是否需要刷新"""
        try:
            # 如果没有access_token或过期时间，需要刷新
            if not credential_data.get("access_token") and not credential_data.get("token"):
                log.debug("没有access_token，需要刷新")
                return True
                
            expiry_str = credential_data.get("expiry")
            if not expiry_str:
                log.debug("没有过期时间，需要刷新")
                return True
                
            # 解析过期时间
            try:
                if isinstance(expiry_str, str):
                    if "+" in expiry_str:
                        file_expiry = datetime.fromisoformat(expiry_str)
                    elif expiry_str.endswith("Z"):
                        file_expiry = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
                    else:
                        file_expiry = datetime.fromisoformat(expiry_str)
                else:
                    log.debug("过期时间格式无效，需要刷新")
                    return True
                    
                # 确保时区信息
                if file_expiry.tzinfo is None:
                    file_expiry = file_expiry.replace(tzinfo=timezone.utc)
                    
                # 检查是否还有至少5分钟有效期
                now = datetime.now(timezone.utc)
                time_left = (file_expiry - now).total_seconds()
                
                log.debug(f"Token剩余时间: {int(time_left/60)}分钟")
                
                if time_left > 300:  # 5分钟缓冲
                    return False
                else:
                    log.debug(f"Token即将过期（剩余{int(time_left/60)}分钟），需要刷新")
                    return True
                    
            except Exception as e:
                log.warning(f"解析过期时间失败: {e}，需要刷新")
                return True
                
        except Exception as e:
            log.error(f"检查token过期时出错: {e}")
            return True
    
    async def _refresh_token(self, credential_data: Dict[str, Any], filename: str) -> Optional[Dict[str, Any]]:
        """刷新token并更新存储"""
        try:
            # 创建Credentials对象
            creds = Credentials.from_dict(credential_data)
            
            # 检查是否可以刷新
            if not creds.refresh_token:
                log.error(f"没有refresh_token，无法刷新: {filename}")
                return None
                
            # 刷新token
            log.debug(f"正在刷新token: {filename}")
            await creds.refresh()
            
            # 更新凭证数据
            if creds.access_token:
                credential_data["access_token"] = creds.access_token
                # 保持兼容性
                credential_data["token"] = creds.access_token
                
            if creds.expires_at:
                credential_data["expiry"] = creds.expires_at.isoformat()
                
            # 保存到存储
            await self._storage_adapter.store_credential(filename, credential_data)
            log.info(f"Token刷新成功并已保存: {filename}")
            
            return credential_data
            
        except Exception as e:
            error_msg = str(e)
            log.error(f"Token刷新失败 {filename}: {error_msg}")
            
            # 检查是否是凭证永久失效的错误
            is_permanent_failure = self._is_permanent_refresh_failure(error_msg)
            
            if is_permanent_failure:
                log.warning(f"检测到凭证永久失效: {filename}")
                # 记录失效状态，但不在这里禁用凭证，让上层调用者处理
                await self.record_api_call_result(filename, False, 400)
            
            return None
    
    def _is_permanent_refresh_failure(self, error_msg: str) -> bool:
        """判断是否是凭证永久失效的错误"""
        # 常见的永久失效错误模式
        permanent_error_patterns = [
            "400 Bad Request",
            "invalid_grant",
            "refresh_token_expired", 
            "invalid_refresh_token",
            "unauthorized_client",
            "access_denied"
        ]
        
        error_msg_lower = error_msg.lower()
        for pattern in permanent_error_patterns:
            if pattern.lower() in error_msg_lower:
                return True
                
        return False

    # 兼容性方法 - 保持与现有代码的接口兼容
    async def _update_token_in_file(self, file_path: str, new_token: str, expires_at=None):
        """更新凭证令牌（兼容性方法）"""
        try:
            credential_data = await self._storage_adapter.get_credential(file_path)
            if not credential_data:
                log.error(f"Credential not found for token update: {file_path}")
                return False
            
            # 更新令牌数据
            credential_data["token"] = new_token
            if expires_at:
                credential_data["expiry"] = expires_at.isoformat() if hasattr(expires_at, 'isoformat') else expires_at
            
            # 保存更新后的凭证
            success = await self._storage_adapter.store_credential(file_path, credential_data)
            
            if success:
                log.debug(f"Token updated for credential: {file_path}")
            else:
                log.error(f"Failed to update token for credential: {file_path}")
            
            return success
            
        except Exception as e:
            log.error(f"Error updating token for {file_path}: {e}")
            return False


# 全局实例管理（保持兼容性）
_credential_manager: Optional[CredentialManager] = None

async def get_credential_manager() -> CredentialManager:
    """获取全局凭证管理器实例"""
    global _credential_manager
    
    if _credential_manager is None:
        _credential_manager = CredentialManager()
        await _credential_manager.initialize()
    
    return _credential_manager