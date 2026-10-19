# AskAnswer

[中文](#中文) · [English](#english)

基于 LangGraph 的可嵌入智能体核心，支持意图路由、工具调用、持久化会话与人工确认。

An embeddable LangGraph agent core with intent routing, tool use, persistent conversations, and human approval.

## 中文

### 项目简介

AskAnswer 将问题理解、工具执行和回答评估组合为一个可嵌入 Python 应用的工作流。宿主应用通过 Python API 提交问题、消费事件，并在需要澄清或确认时收集用户输入。

- **多意图路由**：支持对话、搜索、文件分析、SQL、数学、研究、决策和 Helix 规格推演。
- **工具与子智能体**：内置搜索、天气、时间、计算、汇率、IP 查询和文件工具；研究、SQL、决策与 Helix 使用专门的子图。
- **流式事件**：统一输出文本增量、工具规划、节点更新、中断和最终答案。
- **人工介入**：通过 LangGraph interrupt 暂停澄清流程，以及标记为需要确认的 shell、文件写入和付费外部 API 操作。
- **会话与审计**：SQLite 保存检查点、会话元数据和审计记录，支持恢复会话。
- **可扩展配置**：分层配置、按角色选择模型及回退、上下文裁剪，以及通过 stdio、SSE 或 Streamable HTTP 接入 MCP 工具。

### 安装

需要 Python 3.10 或更高版本。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活虚拟环境。

在项目根目录创建 `.env`，填写所选模型提供方的凭据。默认提供方为 OpenAI，默认模型为 `openai:gpt-5.4`：

```dotenv
OPENAI_API_KEY=your-api-key
ASKANSWER_DEFAULT_MODEL=openai:gpt-5.4

# 使用联网搜索或天气工具时配置
TAVILY_API_KEY=your-tavily-key
OPENWEATHER_API_KEY=your-openweather-key
```

模型名称使用 `provider:model` 格式。切换提供方时，需要安装对应的 LangChain 集成包并配置其凭据。完整变量清单见 [.env.example](.env.example) 与 [settings.example.json](settings.example.json)。

`import askanswer` **没有副作用**：不读配置文件、不修改 `os.environ`、不打开数据库、不需要 API key。`.env` 与 settings.json 会在首次使用时惰性加载（`create_search_assistant()`、首次调用模型、`get_persistence()`、`runtime_context_from_env()` 等）。如果宿主只想用进程环境变量与 `ContextSchema` 管理配置，可在最早处调用 `askanswer.config.settings.bootstrap_environ(enabled=False)` 完全关闭文件配置加载。

### 快速开始

将以下代码保存为 `example.py`，然后运行 `python example.py`：

```python
from askanswer import ContextSchema, create_search_assistant, query_input, run_leg

app = create_search_assistant()
context = ContextSchema()
thread_id = "demo-zh"

for event in run_leg(
    app,
    query_input("你好，请简单介绍一下你能做什么。"),
    thread_id=thread_id,
    context=context,
):
    if event.kind == "final":
        print(event.text)
    elif event.kind == "interrupt":
        print("等待用户输入：", event.data)
```

同一会话继续使用相同的 `thread_id`；新会话使用新的 ID。默认状态数据库为 `~/.askanswer/state.db`，设置 `XDG_DATA_HOME` 后则位于 `$XDG_DATA_HOME/askanswer/state.db`；也可通过 `ASKANSWER_DB_PATH` 指定路径。

### 事件与恢复执行

一次 `run_leg()` 调用从提交问题或恢复执行开始，到中断或完成为止。宿主应用需要遍历返回的生成器来实际执行工作流。

| `event.kind` | 含义 |
| --- | --- |
| `token` | 文本增量，位于 `event.text` |
| `tool` | 开始规划工具调用；应清空当前文本缓冲，规划文本不属于最终答案 |
| `node` | 节点更新，包含 `event.node`、`event.data` 和耗时 |
| `interrupt` | 等待用户输入，载荷位于 `event.data`；本段不会再产生 `final` |
| `final` | 本段完成，最终答案位于 `event.text` |

收到中断后，先向用户展示载荷并收集回应，再使用相同的应用、会话 ID 和上下文调用 `run_leg()`，将输入替换为 `resume_input(decision)`：

| 中断类型（`event.data["type"]`） | `decision` 格式 |
| --- | --- |
| `confirm_shell`、`confirm_fs_write`、`confirm_external_api_paid` | `{"approve": True}` 或 `{"approve": False}` |
| `clarify`：选择已有选项 | `{"index": 0}`，索引从 0 开始 |
| `clarify`：填写自由文本（载荷允许时） | `{"index": len(payload["labels"]), "text": "用户输入"}`，其中 `payload = event.data` |

`resume_input` 可从 `askanswer` 导入。恢复后仍可能再次中断，宿主需继续处理。`node` 事件可能包含 LangChain 消息对象；通过网络传输时需由宿主转换为可序列化的数据。

`run_leg()` 同时处理审计、遥测和会话元数据。如果只需要图事件，可使用 `stream_leg(app, graph_input, config=thread_config(thread_id), context=context)`。向 `create_search_assistant(checkpointer=...)` 传入自定义检查点存储可替换默认的图持久化：`run_leg()` 只在 `app.checkpointer` 属于内置 `PersistenceManager` 时把审计与元数据写入 SQLite，否则（如 `InMemorySaver()`）使用不落盘的 `NullPersistence`；也可以通过 `run_leg(..., persistence=backend)` 显式指定写入后端。

### 配置

配置优先级从高到低为：

1. 进程中已有的环境变量。
2. 项目 `.askanswer/settings.local.json`。
3. 项目 `.askanswer/settings.json`。
4. 用户 `~/.askanswer/settings.json`（可通过 `ASKANSWER_SETTINGS` 或 `XDG_CONFIG_HOME` 改变位置）。
5. `.env`。

例如，在 `.askanswer/settings.json` 中设置：

```json
{
  "model": "openai:gpt-5.4",
  "db_path": "~/.askanswer/state.db",
  "context": {
    "max_tokens": 24000,
    "digest": "brief"
  }
}
```

`models.classify`、`models.evaluate`、`models.summarize` 和 `models.answer` 可分别覆盖角色模型，`models.fallbacks` 可配置各角色的回退列表。上下文裁剪默认关闭；`digest` 支持 `off`、`brief`（确定性摘要）和 `llm`（模型摘要）。`max_tokens` 是基于估算的裁剪目标，会保留系统消息和最新消息。

SQL 工作流通过 `ContextSchema(db_dsn="...", db_dialect="postgresql")` 接收数据库连接配置（SQLAlchemy URL），也可用环境变量 `ASKANSWER_DB_DSN`（或 settings.json 的 `db_dsn`）；旧名 `WLANGGRAPH_POSTGRES_DSN` 仍可读取，但会发出 `DeprecationWarning`。底层直接使用 SQLAlchemy（不再依赖 `langchain-community`），需要安装对应的数据库驱动；SQL agent 只允许只读查询；慢 SQL 统计工具依赖 PostgreSQL 的 `pg_stat_statements`。

### MCP 与扩展

宿主通过 `askanswer.tools.mcp_manager.get_manager()` 获取 MCP 管理器，调用 `add_url()` 或 `add_stdio()` 连接服务，然后调用 `askanswer.tools.get_registry().refresh_mcp()` 同步工具。工具名称采用 `<server>__<tool>` 格式。断开服务或刷新健康状态后，也应重新同步注册表。

MCP 工具默认仅对 `chat` 意图开放；设置 `ASKANSWER_MCP_ALL_INTENTS=1` 可对所有意图开放（每次 `refresh_mcp()` 时读取）。应用退出时调用管理器的 `close()` 释放连接。

新增意图通过 `askanswer.intents.register(handler)` 注册；新增工具通过 `ToolDescriptor` 和 `get_registry().register(...)` 注册。具体接口见 [意图协议](askanswer/intents/base.py) 和 [工具注册表](askanswer/tools/registry.py)。

### 开发

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check askanswer tests
```

可选遥测依赖通过 `python -m pip install -e ".[telemetry]"` 安装（设置 `LANGSMITH_API_KEY` 或 `ASKANSWER_OTEL_EXPORTER` 后首次使用时自动启用）；集成入口见 [observability/telemetry.py](askanswer/observability/telemetry.py)。

| 路径 | 职责 |
| --- | --- |
| [askanswer/core/](askanswer/core/) | 主图（理解 → 回答 → 评估）、运行器（事件流与记账）、状态、上下文裁剪、澄清、确认协议、ReAct 子图 |
| [askanswer/config/](askanswer/config/) | 分层配置与惰性加载、`ContextSchema` |
| [askanswer/llm/](askanswer/llm/) | 可热替换的全局模型代理、按角色路由与回退 |
| [askanswer/intents/](askanswer/intents/) | 意图识别、提示、澄清和评估 |
| [askanswer/tools/](askanswer/tools/) | 工具注册表、内置 / 文件 / shell 工具、MCP 管理与适配 |
| [askanswer/subagents/](askanswer/subagents/) | SQL、Helix、研究、决策子图与共享访谈节点 |
| [askanswer/storage/](askanswer/storage/) | SQLite 检查点、会话元数据、审计、schema 迁移、`NullPersistence` |
| [askanswer/observability/](askanswer/observability/) | 审计缓冲、参数脱敏、LangSmith / OpenTelemetry |
| [tests/](tests/) | 按包镜像的自动化测试 |

架构评审与重构记录见 [docs/design/ARCH_REVIEW_CORE_AGENT.md](docs/design/ARCH_REVIEW_CORE_AGENT.md)，变更记录见 [CHANGELOG.md](CHANGELOG.md)。

## English

### Overview

AskAnswer combines query understanding, tool execution, and answer evaluation into a workflow you can embed in a Python application. The host submits questions through the Python API, consumes events, and collects user input when clarification or approval is needed.

- **Intent routing**: chat, search, file analysis, SQL, math, research, decision support, and Helix specification refinement.
- **Tools and subagents**: built-in search, weather, time, calculation, currency, IP lookup, and file tools; dedicated subgraphs for research, SQL, decisions, and Helix.
- **Streaming events**: a common interface for text deltas, tool planning, node updates, interrupts, and final answers.
- **Human input**: LangGraph interrupts pause clarification and operations marked for approval, including shell commands, file writes, and paid external API calls.
- **Conversations and audit**: SQLite stores checkpoints, thread metadata, and audit records for resumable conversations.
- **Extensibility**: layered settings, model selection and fallbacks by role, context trimming, and MCP tools over stdio, SSE, or Streamable HTTP.

### Installation

Requires Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Create a `.env` file in the project root with credentials for your chosen model provider. The default provider is OpenAI and the default model is `openai:gpt-5.4`:

```dotenv
OPENAI_API_KEY=your-api-key
ASKANSWER_DEFAULT_MODEL=openai:gpt-5.4

# Configure these when using web search or weather tools
TAVILY_API_KEY=your-tavily-key
OPENWEATHER_API_KEY=your-openweather-key
```

Model names use the `provider:model` format. Other providers require the corresponding LangChain integration package and credentials. See [.env.example](.env.example) and [settings.example.json](settings.example.json) for every supported variable.

`import askanswer` has **no side effects**: it reads no configuration files, does not modify `os.environ`, opens no database and needs no API key. `.env` and settings files are loaded lazily on first use (`create_search_assistant()`, the first model call, `get_persistence()`, `runtime_context_from_env()`, …). Hosts that manage configuration purely through process environment variables and `ContextSchema` can call `askanswer.config.settings.bootstrap_environ(enabled=False)` early to opt out of file-based configuration.

### Quick start

Save this as `example.py`, then run `python example.py`:

```python
from askanswer import ContextSchema, create_search_assistant, query_input, run_leg

app = create_search_assistant()
context = ContextSchema()
thread_id = "demo-en"

for event in run_leg(
    app,
    query_input("Hello, briefly introduce what you can help with."),
    thread_id=thread_id,
    context=context,
):
    if event.kind == "final":
        print(event.text)
    elif event.kind == "interrupt":
        print("Waiting for user input:", event.data)
```

Reuse `thread_id` to continue a conversation; use a new ID for a separate conversation. The default state database is `~/.askanswer/state.db`, or `$XDG_DATA_HOME/askanswer/state.db` when `XDG_DATA_HOME` is set. Override it with `ASKANSWER_DB_PATH`.

### Events and resuming

Each `run_leg()` call runs from a new question or a resume decision until an interrupt or completion. The host must iterate over the returned generator to execute the workflow.

| `event.kind` | Meaning |
| --- | --- |
| `token` | Text delta in `event.text` |
| `tool` | Tool planning has started; clear the current text buffer because planning text is not part of the final answer |
| `node` | Node update with `event.node`, `event.data`, and elapsed time |
| `interrupt` | User input is required; payload is in `event.data`, and this leg emits no `final` |
| `final` | The leg completed; the final answer is in `event.text` |

After an interrupt, display the payload and collect the user's response. Call `run_leg()` again with the same app, thread ID, and context, replacing the graph input with `resume_input(decision)`:

| Interrupt type (`event.data["type"]`) | `decision` format |
| --- | --- |
| `confirm_shell`, `confirm_fs_write`, `confirm_external_api_paid` | `{"approve": True}` or `{"approve": False}` |
| `clarify`: choose an existing option | `{"index": 0}`, using a zero-based index |
| `clarify`: enter free text, when allowed by the payload | `{"index": len(payload["labels"]), "text": "user response"}`, where `payload = event.data` |

Import `resume_input` from `askanswer`. A resumed leg can interrupt again, so the host must keep handling these events. Node payloads may contain LangChain message objects; the host must convert them to serializable data before network transport.

`run_leg()` also handles audit, telemetry, and thread metadata. For graph events alone, use `stream_leg(app, graph_input, config=thread_config(thread_id), context=context)`. Pass a custom checkpointer to `create_search_assistant(checkpointer=...)` to replace graph persistence. `run_leg()` writes audit and thread metadata to SQLite only when `app.checkpointer` belongs to a built-in `PersistenceManager`; otherwise (e.g. `InMemorySaver()`) it uses `NullPersistence` and nothing touches disk. Pass `run_leg(..., persistence=backend)` to choose the backend explicitly.

### Configuration

Configuration precedence, highest first:

1. Existing process environment variables.
2. Project `.askanswer/settings.local.json`.
3. Project `.askanswer/settings.json`.
4. User `~/.askanswer/settings.json` (location can be changed through `ASKANSWER_SETTINGS` or `XDG_CONFIG_HOME`).
5. `.env`.

For example, in `.askanswer/settings.json`:

```json
{
  "model": "openai:gpt-5.4",
  "db_path": "~/.askanswer/state.db",
  "context": {
    "max_tokens": 24000,
    "digest": "brief"
  }
}
```

Override individual model roles with `models.classify`, `models.evaluate`, `models.summarize`, and `models.answer`; use `models.fallbacks` for per-role fallback lists. Context trimming is disabled by default. Digest modes are `off`, `brief` (deterministic summary), and `llm` (model-generated summary). `max_tokens` is an estimated trimming target that preserves system messages and the latest message.

SQL workflows accept database configuration through `ContextSchema(db_dsn="...", db_dialect="postgresql")` (a SQLAlchemy URL), or the `ASKANSWER_DB_DSN` environment variable (`db_dsn` in settings.json). The legacy `WLANGGRAPH_POSTGRES_DSN` is still read but emits a `DeprecationWarning`. The SQL subagent uses SQLAlchemy directly (no `langchain-community`); install the appropriate database driver. Only read-only queries are executed. Slow-query statistics require PostgreSQL's `pg_stat_statements`.

### MCP and extensions

Get the MCP manager through `askanswer.tools.mcp_manager.get_manager()`, connect a server with `add_url()` or `add_stdio()`, then synchronize tools with `askanswer.tools.get_registry().refresh_mcp()`. Tool names use `<server>__<tool>`. Refresh the registry after disconnecting a server or updating its health status as well.

MCP tools are exposed to the `chat` intent by default. Set `ASKANSWER_MCP_ALL_INTENTS=1` to expose them to all intents (read on every `refresh_mcp()`). Call the manager's `close()` when your application shuts down.

Register new intents with `askanswer.intents.register(handler)` and tools with `ToolDescriptor` and `get_registry().register(...)`. See the [intent protocol](askanswer/intents/base.py) and [tool registry](askanswer/tools/registry.py) for the interfaces.

### Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check askanswer tests
```

Install optional telemetry dependencies with `python -m pip install -e ".[telemetry]"`; exporters turn on at first use when `LANGSMITH_API_KEY` or `ASKANSWER_OTEL_EXPORTER` is set. See [observability/telemetry.py](askanswer/observability/telemetry.py) for integration points.

| Path | Responsibility |
| --- | --- |
| [askanswer/core/](askanswer/core/) | Main graph (understand → answer → evaluate), runner (events and bookkeeping), state, context budget, clarification, confirmation protocol, ReAct subgraph |
| [askanswer/config/](askanswer/config/) | Layered settings with lazy bootstrap, `ContextSchema` |
| [askanswer/llm/](askanswer/llm/) | Hot-swappable global model proxy, role routing and fallbacks |
| [askanswer/intents/](askanswer/intents/) | Intent classification, prompts, clarification, and evaluation |
| [askanswer/tools/](askanswer/tools/) | Tool registry, built-in / filesystem / shell tools, MCP manager and adapter |
| [askanswer/subagents/](askanswer/subagents/) | SQL, Helix, research and decision subgraphs plus the shared interview node |
| [askanswer/storage/](askanswer/storage/) | SQLite checkpoints, thread metadata, audit, schema migrations, `NullPersistence` |
| [askanswer/observability/](askanswer/observability/) | Audit buffer, argument redaction, LangSmith / OpenTelemetry |
| [tests/](tests/) | Automated tests mirroring the package layout |

The architecture review and refactoring record is in [docs/design/ARCH_REVIEW_CORE_AGENT.md](docs/design/ARCH_REVIEW_CORE_AGENT.md); see [CHANGELOG.md](CHANGELOG.md) for changes.
