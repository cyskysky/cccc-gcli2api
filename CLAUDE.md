# CLAUDE.md

该文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 常用命令

### 安装

要设置开发环境，您可以使用适用于您操作系统的安装脚本（例如 `install.sh`, `install.ps1`, `darwin-install.sh`）。

或者，您可以手动安装依赖项：

```bash
pip install -r requirements.txt
```

### 运行服务

要启动服务器，请运行 `web.py` 文件：

```bash
python web.py
```

您也可以使用适用于您平台的启动脚本（例如 `start.sh`, `start.bat`）。

服务器默认将在 `http://127.0.0.1:7861` 上启动。

### 运行测试

该项目不使用像 `pytest` 这样的标准测试运行器。相反，测试作为独立的 Python 脚本运行。要运行一个测试，请直接执行测试文件：

```bash
python test_tool_calling.py
```

## 架构概览

该项目是一个代理服务器，可将 Gemini CLI 调用转换为与 OpenAI 和 Gemini API 兼容的格式。它使用 FastAPI 构建，并在 Hypercorn 上异步运行。

### 入口点

应用程序的主入口点是 `web.py`。该文件初始化 FastAPI 应用程序，设置中间件，并包含主路由。它还管理应用程序的生命周期，包括 `CredentialManager` 的初始化和关闭。

### 核心组件

应用程序的核心逻辑位于 `src/` 目录中。关键模块包括：

*   **路由 (`src/openai_router.py`, `src/gemini_router.py`, `src/web_routes.py`):** 应用程序分为三个主路由：
    *   `openai_router.py`: 处理与 OpenAI 兼容的 API 端点。
    *   `gemini_router.py`: 处理 Gemini 原生的 API 端点。
    *   `web_routes.py`: 管理 Web 控制面板，包括身份验证、凭证管理和配置。

*   **凭证管理 (`src/credential_manager.py`):** 这是一个核心组件，负责管理 OAuth 凭证，包括轮换、状态跟踪和容错。

*   **API 转换 (`src/openai_transfer.py`, `src/format_detector.py`):** 这些模块负责在 OpenAI 和 Gemini 格式之间转换请求和响应。`format_detector.py` 自动检测传入的请求格式，`openai_transfer.py` 处理转换逻辑。

*   **状态与存储 (`src/state_manager.py`, `src/usage_stats.py`, `src/storage_adapter.py`):**
    *   `state_manager.py` 和 `usage_stats.py` 处理应用程序的运行时状态和使用情况统计。
    *   `storage_adapter.py` 为不同的存储后端（Redis、Postgres、MongoDB 和本地文件）提供了一个抽象层，允许应用程序在各种环境中部署。

### 配置

配置通过 `config.py` 进行管理，它提供了一个用于检索配置值的层级系统。优先级顺序如下：

1.  环境变量
2.  存储后端（例如 Redis, MongoDB）
3.  `config.py` 中定义的默认值

这允许在不同的部署场景中进行灵活的配置。

### 前端

Web 控制面板的前端位于 `front/` 目录中。`src/web_routes.py` 模块根据用户的用户代理（user agent）提供相应的 HTML 文件（桌面版为 `control_panel.html`，移动版为 `control_panel_mobile.html`）。
