# AskAnswer

基于 LangGraph 的命令行智能助手：按意图分流到「读本地文件 / 联网搜索 / SQL 查询 / 调研简报 / 决策备忘 / 规格演化 / 直接回答」，由 LLM 调度工具并整合结果，支持 HITL 确认、会话持久化、时间旅行与 Markdown 渲染。

当前状态（2026-07-16）：分层 `settings.json`（类似 Claude Code）+ `.env` 保底；`pytest` 全绿；GitHub Actions 对 Python 3.10 / 3.12 跑 ruff + pytest。CLI 已拆成 `askanswer/cli/` 包，与 HTTP/SSE 服务共用 `runner` 事件流。

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # Python >= 3.10
pip install -e ".[dev]"     # + pytest / ruff
```

## 配置

配置由 `load.py` → `settings.bootstrap_environ()` 在导入时装载。支持 **settings.json**（推荐）与 **`.env`**（保底）并存；运行时一切仍落成环境变量，其余代码只读 `os.environ`。

### 优先级（高 → 低）

| 顺序 | 来源 | 说明 |
| --- | --- | --- |
| 1 | **进程环境变量** | shell / CI `export`；启动前已存在的键**永不被文件覆盖** |
| 2 | **Local settings** | `<repo>/.askanswer/settings.local.json`（个人，已 gitignore） |
| 3 | **Project settings** | `<repo>/.askanswer/settings.json`（可提交，团队共享） |
| 4 | **User settings** | `~/.askanswer/settings.json`（跨项目；见路径说明） |
| 5 | **`.env`** | 项目根目录；**最低保底**，只填补仍空缺的变量 |

> 实现要点：先快照进程已有键 → `load_dotenv(override=False)` → settings 以 `override=True` 盖过 `.env`，但 `protect` 快照键。损坏或缺失的 JSON 静默忽略，不阻塞启动。

### 快速上手

**方式 A — 只用 `.env`（最简单）**

```bash
cp .env.example .env
# 编辑 OPENAI_API_KEY / TAVILY_API_KEY 等
```

**方式 B — settings.json（推荐结构化配置）**

```bash
mkdir -p ~/.askanswer
cp settings.example.json ~/.askanswer/settings.json
# 或项目级：mkdir -p .askanswer && cp settings.example.json .askanswer/settings.json
# 个人覆盖（勿提交）：cp settings.example.json .askanswer/settings.local.json
```

密钥可放在 settings 的 `env` 块，也可继续放在 `.env` 做保底；进程里 `export` 的值始终最高。

### settings.json（类似 Claude Code）

#### 路径

| 作用域 | 默认路径 | 覆盖方式 |
| --- | --- | --- |
| User | `~/.askanswer/settings.json` | `$XDG_CONFIG_HOME/askanswer/settings.json`，或 `ASKANSWER_SETTINGS=/path/to.json` |
| Project | `<repo>/.askanswer/settings.json` | 从 cwd 向上找 `.git` / `pyproject.toml` / `.askanswer` |
| Local | `<repo>/.askanswer/settings.local.json` | 同 project；gitignored |

完整模板：[`settings.example.json`](settings.example.json)。

#### 示例

```json
{
  "model": "openai:gpt-5.4",
  "models": {
    "classify": "openai:gpt-4o-mini",
    "evaluate": "openai:gpt-4o-mini",
    "summarize": "openai:gpt-4o-mini",
    "fallbacks": {
      "answer": ["openai:gpt-4o", "deepseek:deepseek-chat"]
    }
  },
  "context": { "max_tokens": 24000, "digest": "brief" },
  "run_token_budget": 60000,
  "tenant_id": "alice",
  "db_path": "~/.askanswer/state.db",
  "mcp_all_intents": false,
  "env": {
    "OPENAI_API_KEY": "sk-...",
    "TAVILY_API_KEY": "...",
    "OPENWEATHER_API_KEY": ""
  }
}
```

#### 一等字段 → 环境变量

| JSON 字段 | 展开为 | 说明 |
| --- | --- | --- |
| `model` | `ASKANSWER_DEFAULT_MODEL` | 启动默认模型（`/model` 仍可热切换） |
| `models.classify` / `.evaluate` / `.summarize` / `.answer` | `ASKANSWER_MODEL_<ROLE>` | 按角色路由 |
| `models.fallbacks.<role>` | `ASKANSWER_MODEL_FALLBACKS_<ROLE>` | 数组会拼成逗号分隔列表 |
| `context.max_tokens` | `ASKANSWER_CONTEXT_MAX_TOKENS` | answer 历史 token 预算 |
| `context.digest` | `ASKANSWER_CONTEXT_DIGEST` | `brief` 或 `llm` |
| `run_token_budget` | `ASKANSWER_RUN_TOKEN_BUDGET` | 单轮 token 上限 |
| `tenant_id` / `tenantId` | `ASKANSWER_TENANT_ID` | 多租户隔离 |
| `db_path` / `dbPath` | `ASKANSWER_DB_PATH` | SQLite 状态库路径 |
| `db_dialect` / `dbDialect` | `ASKANSWER_DB_DIALECT` | SQL agent 方言 |
| `mcp_profile` / `mcpProfile` | `ASKANSWER_MCP_PROFILE` | MCP 自动连接 profile |
| `mcp_all_intents` / `mcpAllIntents` | `ASKANSWER_MCP_ALL_INTENTS` | bool → `1` / `0` |
| `server_token` / `serverToken` | `ASKANSWER_SERVER_TOKEN` | HTTP Bearer |
| `env` | 任意键原样写入 | **同名时覆盖**上方一等字段 |

### `.env`（最低保底）

项目根目录创建 `.env`（参考 [`.env.example`](.env.example)）。仅在变量尚未被进程或 settings 设置时生效：

```
OPENAI_API_KEY=...
OPENAI_BASE_URL=...           # 可选
TAVILY_API_KEY=...            # 联网搜索
OPENWEATHER_API_KEY=...       # 可选，天气工具
WLANGGRAPH_POSTGRES_DSN=...   # 可选，SQL agent 默认库
ASKANSWER_DB_DIALECT=         # 可选；留空则从连接推断
ASKANSWER_DB_PATH=            # 可选；默认 ~/.askanswer/state.db
ASKANSWER_TENANT_ID=          # 可选；多租户
ASKANSWER_SERVER_TOKEN=       # 可选；HTTP 鉴权
ASKANSWER_DEFAULT_MODEL=      # 可选；默认 openai:gpt-5.4
```

### 模型路由与成本控制（可选，默认关闭）

按角色给不同任务配不同模型：分类 / 评估 / 摘要用小模型，主回答可配跨厂商回退链。未设置时与单模型模式完全一致（`/model` 语义不变）。

**推荐写在 settings.json**（见上表 `models` / `context` / `run_token_budget`），等价环境变量：

```
ASKANSWER_MODEL_CLASSIFY=openai:gpt-4o-mini
ASKANSWER_MODEL_EVALUATE=openai:gpt-4o-mini
ASKANSWER_MODEL_SUMMARIZE=openai:gpt-4o-mini
ASKANSWER_MODEL_FALLBACKS_ANSWER=openai:gpt-4o,deepseek:deepseek-chat
ASKANSWER_CONTEXT_MAX_TOKENS=24000
ASKANSWER_CONTEXT_DIGEST=brief
ASKANSWER_RUN_TOKEN_BUDGET=60000
```

- 回退写 `model_fallback` 审计事件；token 归因到**真正执行**的模型；`/status` 显示非默认路由，`/usage` 按标签分账（价格表见 `askanswer/pricing.py`）。
- system prompt 按「稳定前缀 → 动态尾部」排序以命中 prompt cache；answer 指向 `anthropic:*` 时自动加 `cache_control`。
- 离线评测：`python evals/run_intent_eval.py`（见 `evals/README.md`）；设计说明：`docs/important-documentation-d1-routing-context-cost-eval.md`。

### 其他常用环境变量

| 变量 | 说明 |
| --- | --- |
| `ASKANSWER_SETTINGS` | 覆盖 **user** 层 settings.json 路径 |
| `ASKANSWER_MCP_ALL_INTENTS` | `1` 时 MCP 工具对所有意图可见（默认仅 chat） |
| `ASKANSWER_READ_FILE_MAX_BYTES` | `read_file` 上限（默认 10 MB） |
| `ASKANSWER_WRITE_FILE_MAX_BYTES` | `write_file` 上限（默认 1 MB） |
| `ASKANSWER_SHELL_OUTPUT_MAX_BYTES` | shell 每路输出上限（默认 64 KB） |
| `LANGSMITH_API_KEY` / `ASKANSWER_OTEL_EXPORTER` | 可选可观测导出（关则零开销） |

### 库 API 与 runtime context

SQL agent 等从 LangGraph runtime context 读库配置（不写进 `SearchState`）：

```python
from askanswer.graph import create_search_assistant
from askanswer.schema import ContextSchema

app = create_search_assistant()
app.invoke(
    {"messages": [{"role": "user", "content": "统计一下订单数量"}]},
    context=ContextSchema(db_dsn="postgresql://user:password@localhost:5432/dbname"),
)
```

CLI / HTTP 会把环境中的 `WLANGGRAPH_POSTGRES_DSN` / `ASKANSWER_DB_DIALECT` / `ASKANSWER_TENANT_ID` 注入 context（`runner.runtime_context_from_env()`），来源可以是 settings、`.env` 或进程 export。

## 使用

```bash
askanswer "上海今天天气怎么样"   # 单次提问
askanswer                         # 交互 REPL
python -m askanswer               # 等价入口
askanswer --graph                 # 输出主图 Mermaid
askanswer --graph docs/graph.mmd  # 写入文件（不创建 SQLite）
```

### HTTP / SSE 服务

与 CLI 共用同一图、checkpointer 与 `runner` 事件流（零额外依赖，stdlib only）：

```bash
python -m askanswer.server        # 默认 127.0.0.1:8765
# 或 reinstall 后：askanswer-server
```

| 端点 | 说明 |
| --- | --- |
| `GET /health` | 健康检查 |
| `POST /v1/query` | 提问，SSE 推送 `token` / `tool` / `node` / `interrupt` / `final` |
| `POST /v1/resume` | HITL 续跑（decision 形态与 CLI 一致） |
| `GET /v1/interrupt?thread_id=` | 查询挂起的 interrupt |

安全基线：默认只绑 localhost；可选 Bearer token；Origin + JSON Content-Type 双闸；64KB body 上限；全局 2 路并发 + 同 thread 互斥。契约见 `docs/important-documentation-c3-http-sse-server.md`。

### 交互模式界面

启动后进入类 Claude Code 的 REPL：圆角欢迎框、箱式输入、进度标记 `⏺ Node(detail)`，回答以 Markdown 渲染（重定向到文件时退化为纯文本）。输入历史在 `~/.askanswer/history`（自动过滤 API key / DSN 等敏感模式）。

#### 斜线命令

| 命令 | 说明 |
| --- | --- |
| `/help [cmd]` | 显示帮助 |
| `/clear` | 清屏并开启新会话 |
| `/status` | 当前会话（含 MCP / model） |
| `/model [provider:name]` | 查看或热切换模型 |
| `/mcp …` | 管理 MCP（见下） |
| `/threads [关键词]` | 列出历史会话 |
| `/resume <序号\|id>` | 恢复指定会话 |
| `/title <名字>` | 给当前会话命名 |
| `/delete <序号\|id>` | 删除会话 |
| `/checkpoints` | 列出当前会话快照 |
| `/undo [n] [--label NAME]` | 回退；可命名还原点 |
| `/jump <index>` | 跳到指定快照 |
| `/fork [index]` | 从快照分叉新会话 |
| `/audit …` | 审计事件 |
| `/usage …` | token / 工具用量与费用估算 |
| `/export` / `/import` | 会话导出（md/json）/ 导入 |
| `/edit <path>` | 用系统编辑器打开文件（`$ASKANSWER_EDITOR` / `$VISUAL` / `$EDITOR` → nano/vim…） |
| `/exit` / `/quit` / `/q` | 退出（也可 Ctrl-D） |

快捷前缀 `!<cmd>`：在 REPL 中直接执行 shell（如 `!ls -la`）；命中高风险模式时二次确认。`!vim` / `!nano` 等全屏程序自动 TTY 直通，可直接编辑文件；也可用 `/edit <path>`。

#### MCP

```text
/mcp https://example.com/mcp            # 连接（自动推导服务名；路径以 /sse 结尾走 SSE）
/mcp https://example.com/mcp my-server  # 指定服务名
/mcp add_stdio <name> <cmd> [args…]     # 子进程 stdio MCP
/mcp list [-v]                          # 列表（-v 含健康详情）
/mcp health [name]                      # 健康探测并刷新工具
/mcp tools [server]                     # 列出工具
/mcp remove <name>                      # 断开
```

连接会写入 `~/.askanswer/mcp.json`，下次启动自动重连。多 server 工具以 `<server>__<tool>` 暴露；`call_tool` 按前缀路由。每个 server 的上下文管理器在同一 asyncio 任务内进入/退出，避免 anyio cancel-scope 跨任务问题。

### Helix 规格演化

`helix_spec_loop`：`interview → seed → execute → evaluate → (seed 重跑 | finalize)`。模糊需求（「用苏格拉底问我…」「spec-first…」「需求澄清…」等）命中 `helix` 意图后调用：

- **interview**：scope / constraints / outputs / verification 四轨关键问题并自答（标 `assumption:`）
- **seed** → **execute** → **evaluate**（覆盖率 + 0–1 对齐分 + gaps）
- rejected 时最多 `MAX_GENERATIONS=3` 代回到 seed 修补

返回 Markdown：`## Goal / Constraints / Acceptance criteria / Artifact / Evaluation / Lineage`。也可单独当 MCP 服务：`python -m askanswer.helix_mcp`。设计见 `docs/helix-subgraph-plan.md`。

### 澄清协议（C2）

react 子图入口有 `clarify` 节点：意图 handler 可在首轮回答前 `interrupt` 一次（路径缺失、无 DSN、调研话题过短等）。CLI 用箭头菜单 + 可选自由文本；非 TTY 取默认「不改动」选项，行为非回归。

### HITL 确认

三类确认共用 `confirmations.py` 四步协议（plan → gate → interrupt → apply）：

| 类 | 典型工具 | CLI 提示 |
| --- | --- | --- |
| `shell` | `gen_shell_commands_run` | 执行 / 取消 / 重生成 / 编辑 |
| `fs_write` | `write_file` | diff 预览后写入 / 取消 |
| `external_api_paid` | 付费外部 API | 参数 + 费用警告；审计会脱敏密钥与邮箱 |

## 结构

```
askanswer/
├── cli/                 # REPL 包：app / stream / repl / confirm / render / commands/*
├── graph.py             # 父图：understand → answer(react) → sorcery
├── react.py             # react 子图拓扑（clarify → answer ⇄ tools / confirm_plan）
├── _react_internals.py  # answer / tools / confirm_plan 节点实现
├── clarify.py           # 意图澄清 interrupt
├── confirmations.py     # HITL 确认协议（shell / fs_write / external_api_paid）
├── nodes.py             # 父图节点：意图识别、自评回写
├── state.py / schema.py # SearchState + ContextSchema
├── tools.py / registry.py
├── runner.py / wire.py / server.py   # UI-free 事件流 + HTTP/SSE
├── persistence.py / timetravel.py / audit.py / pricing.py
├── mcp.py / mcp_profile.py / helix_mcp.py
├── telemetry/           # LangSmith + OTEL（env 门控）
├── intents/             # chat / search / file_read / sql / math / helix / research / decision
├── sqlagent/ / helix/ / research/ / decision/   # 子图 → 注册为工具
├── load.py              # 模型代理（/model 热切换）、Tavily、配置 bootstrap
├── settings.py          # settings.json 分层 + .env 保底（bootstrap_environ）
└── ui_input.py / ui_select.py / ui_spinner.py
```


## 工作流

```
START → understand → answer(react) → sorcery → {answer 重跑 | END}

react 子图:
  START → clarify → answer ⇄ tools
                      └→ confirm_plan → tools
```

- **understand**：本地分类器优先（关键词/正则，handler 按 priority），歧义时才调 LLM；提取路径或搜索词
- **clarify**：仅首轮（`step == "understood"`）；handler 可选提问，默认无开销透传
- **answer**：按 `intent` 选系统提示与工具集（`registry.list(tags=…)`）；每轮 `_reclassify_intent` 支持中途换话题
- **tools / confirm_plan**：普通工具直接跑；需确认的先 `plan` 写入 `pending_confirmations`，再 `interrupt()` 等人批准
- **sorcery**：委托 `handler.evaluate`；重试预算按意图（search=2，sql/file_read=1，chat/math/helix/research/decision=0）

## 意图与工具

意图经 `IntentRegistry` 注册；工具经 `ToolRegistry` 按 tag 过滤。内置工具默认对多意图可见；shell 不进 SQL bundle。

| 工具 | 说明 |
| --- | --- |
| `tavily_search` | Tavily 联网搜索 Top 5 + 综合答案 |
| `read_file` | `markitdown` 解析任意文件后由 LLM 总结 |
| `sql_query` | 自然语言 → SQL agent 子图（DSN/方言/租户来自 context） |
| `research_brief_loop` | 调研简报：规划查询 → 搜索 → 综合 → 来源核验 |
| `decision_memo_loop` | 决策备忘：复用 Helix interview + decide |
| `helix_spec_loop` | 规格优先演化（最多 3 代） |
| `check_weather` | OpenWeather 实时天气 |
| `get_current_time` | 指定时区当前时间 |
| `calculate` | 安全表达式计算 |
| `convert_currency` | 货币汇率换算 |
| `lookup_ip` | IP 地理与运营商 |
| `pwd` | 当前工作目录 |
| `gen_shell_commands_run` | 生成并执行 shell；高风险拦截 + HITL |
| `write_file` | 写文件（敏感路径拦截、大小上限、diff 预览）+ `fs_write` 确认 |

MCP 工具以 `<server>__<tool>` 自动入表。

## 持久化与可观测

- **状态 DB**：默认 `~/.askanswer/state.db`（`SqliteSaver` + `thread_meta` + `audit_event`，schema 自迁移；支持 `tenant_id`）
- **时间旅行**：`/checkpoints` `/undo` `/jump` `/fork`（有挂起确认时拒绝 rewind）
- **审计 / 用量**：LLM token、工具调用写入 `audit_event`；`/usage` 可估算费用（`pricing.py`）
- **Telemetry**：`telemetry/` 可选 LangSmith / OpenTelemetry，默认关闭

## 开发与验证

```bash
pytest -q                       # 全量（无需 API key；conftest 隔离 env + 临时 DB）
ruff check askanswer tests      # lint 基线 E/F/I
python -m compileall askanswer
python -m askanswer --graph -   # 图拓扑可编译
```

CI：`.github/workflows/ci.yml` 在 push/PR 到 `master` 时对 3.10 / 3.12 跑 ruff + pytest。

更细的架构索引见 `CLAUDE.md` / `AGENTS.md` → `.claude/mem/*`；变更日志见 `CHANGELOG.md`；待办见 `TODO.md`。
