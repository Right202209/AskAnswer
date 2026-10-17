# AskAnswer

基于 LangGraph 的命令行智能助手。一句话进来，先判断意图（读文件 / 联网搜索 / SQL / 调研简报 / 决策备忘 / 规格演化 / 闲聊），再由 LLM 调度对应工具完成回答。

- **意图分流**：本地规则优先，歧义时才调小模型分类；对话中途可换话题
- **人在回路**：shell、写文件、付费 API 执行前必须确认，恢复时不会重复执行
- **会话持久化**：SQLite 保存全部会话，支持撤销、跳转、分叉（时间旅行）
- **多入口**：CLI REPL、HTTP/SSE 服务、Python 库、MCP 服务（Helix）
- **成本可控**：按角色路由模型、跨厂商回退、单轮 token 预算、用量与费用统计
- **可扩展**：意图、工具、MCP server 都按注册表接入

> 状态（2026-10-17）：架构重构完成，主图为单层 7 节点，依赖经组装根注入；227 项测试全绿。重构细节与新旧模块对照见 [`docs/design/ARCH_REFACTOR_STATUS.md`](docs/design/ARCH_REFACTOR_STATUS.md)。

---

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # Python >= 3.10；开发用 pip install -e ".[dev]"
cp .env.example .env             # 填 OPENAI_API_KEY，联网搜索再填 TAVILY_API_KEY

askanswer "上海今天天气怎么样"   # 单次提问
askanswer                         # 交互 REPL（等价 python -m askanswer）
```

---

## 使用

### CLI

```bash
askanswer [问题]                  # 带问题为单次模式，不带进入 REPL
askanswer --graph                 # 输出主图 Mermaid
askanswer --graph docs/graph.mmd  # 写入文件（不会创建 SQLite）
```

REPL 风格类似 Claude Code：Markdown 渲染、`⏺ Node(detail)` 进度标记、输入历史存于 `~/.askanswer/history`（自动过滤密钥 / DSN）。

| 分类 | 命令 |
| --- | --- |
| 基础 | `/help [cmd]` · `/clear` · `/status` · `/model [provider:name]` · `/exit`（或 Ctrl-D） |
| 会话 | `/threads [关键词]` · `/resume <序号\|id>` · `/title <名字>` · `/delete <序号\|id>` · `/export` · `/import` |
| 时间旅行 | `/checkpoints` · `/undo [n] [--label NAME]` · `/jump <index>` · `/fork [index]` |
| 审计 / 用量 | `/audit …` · `/usage …`（token、工具次数、费用估算） |
| 编辑 / shell | `/edit <path>` · `!<cmd>`（高风险命令二次确认；`!vim` 等自动 TTY 直通） |
| MCP | `/mcp …`（见下） |

```text
/mcp https://example.com/mcp [name]     # 连接 HTTP MCP（路径以 /sse 结尾走 SSE）
/mcp add_stdio <name> <cmd> [args…]     # 子进程 stdio MCP
/mcp list [-v] | health [name] | tools [server] | remove <name>
```

MCP 连接写入 `~/.askanswer/mcp.json` 并在下次启动时自动重连；工具以 `<server>__<tool>` 命名，默认只对 chat 意图可见（`ASKANSWER_MCP_ALL_INTENTS=1` 放开）。

### HTTP / SSE 服务

与 CLI 共用同一张图、同一个 checkpointer 和同一条事件流，纯标准库实现。

```bash
askanswer-server                                 # 或 python -m askanswer.interfaces.http.server
                                                 # 默认 127.0.0.1:8765
```

| 端点 | 说明 |
| --- | --- |
| `GET /health` | 健康检查 |
| `POST /v1/query` | 提问；SSE 推送 `token` / `tool` / `node` / `interrupt` / `final` |
| `POST /v1/resume` | 回复确认 / 澄清后继续（decision 格式与 CLI 一致） |
| `GET /v1/interrupt?thread_id=` | 查询挂起中的确认 |

默认只绑 localhost；可选 Bearer token（`ASKANSWER_SERVER_TOKEN`）；校验 Origin 与 JSON Content-Type；请求体上限 64 KB；全局 2 路并发，同一 thread 互斥。契约见 [`docs/important-documentation-c3-http-sse-server.md`](docs/important-documentation-c3-http-sse-server.md)。

### 作为 Python 库

```python
from askanswer.graph import create_search_assistant
from askanswer.graph.context import ContextSchema

app = create_search_assistant()   # 默认持久化到 ~/.askanswer/state.db
app.invoke(
    {"messages": [{"role": "user", "content": "统计一下订单数量"}]},
    config={"configurable": {"thread_id": "demo"}},
    context=ContextSchema(db_dsn="postgresql://user:password@localhost:5432/dbname"),
)
```

数据库连接、方言、租户等运行期参数走 LangGraph runtime context，不进入会话状态。CLI / HTTP 通过 `askanswer.runtime.runner.runtime_context_from_env()` 从环境变量自动填充。测试时可传入 `container=build_container(...)` 替换模型、存储等依赖。

---

## 配置

入口（CLI / HTTP / MCP server）启动时调用 `askanswer.config.ensure_environ()` 装载配置；导入模块本身不会改动环境变量。分层合并后的结果只在 `askanswer/config/loader.py` 解析为类型化 `Settings`，业务代码只读 `Settings`。

### 来源与优先级（高 → 低）

| # | 来源 | 位置 |
| --- | --- | --- |
| 1 | 进程环境变量 | shell / CI 的 `export`，**永不被文件覆盖** |
| 2 | Local settings | `<repo>/.askanswer/settings.local.json`（个人，已 gitignore） |
| 3 | Project settings | `<repo>/.askanswer/settings.json`（可提交，团队共享） |
| 4 | User settings | `~/.askanswer/settings.json`（或 `$XDG_CONFIG_HOME/askanswer/settings.json`、`$ASKANSWER_SETTINGS`） |
| 5 | `.env` | 项目根目录，只补仍空缺的变量 |

`<repo>` 从当前目录向上查找 `.git` / `pyproject.toml` / `.askanswer`。JSON 损坏或缺失会被静默忽略。完整模板见 [`settings.example.json`](settings.example.json)。

### settings.json 示例

```json
{
  "model": "openai:gpt-5.4",
  "models": {
    "classify": "openai:gpt-4o-mini",
    "evaluate": "openai:gpt-4o-mini",
    "summarize": "openai:gpt-4o-mini",
    "fallbacks": { "answer": ["openai:gpt-4o", "deepseek:deepseek-chat"] }
  },
  "context": { "max_tokens": 24000, "digest": "brief" },
  "run_token_budget": 60000,
  "tenant_id": "alice",
  "db_path": "~/.askanswer/state.db",
  "env": {
    "OPENAI_API_KEY": "sk-...",
    "TAVILY_API_KEY": "..."
  }
}
```

### 配置项一览

| settings.json 字段 | 环境变量 | 说明 |
| --- | --- | --- |
| `env.*` | 原样写入 | 任意变量，如 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `TAVILY_API_KEY` / `OPENWEATHER_API_KEY`；同名时覆盖下方字段 |
| `model` | `ASKANSWER_DEFAULT_MODEL` | 默认模型，默认 `openai:gpt-5.4`；`/model` 可热切换 |
| `models.<role>` | `ASKANSWER_MODEL_<ROLE>` | 按角色路由：`classify` / `evaluate` / `summarize` / `answer` |
| `models.fallbacks.<role>` | `ASKANSWER_MODEL_FALLBACKS_<ROLE>` | 回退链（环境变量用逗号分隔） |
| `context.max_tokens` | `ASKANSWER_CONTEXT_MAX_TOKENS` | 回答时带入的历史 token 预算 |
| `context.digest` | `ASKANSWER_CONTEXT_DIGEST` | 超预算历史的压缩方式：`brief` / `llm` |
| `run_token_budget` | `ASKANSWER_RUN_TOKEN_BUDGET` | 单轮 token 上限 |
| `db_path` | `ASKANSWER_DB_PATH` | 状态库路径，默认 `~/.askanswer/state.db` |
| `tenant_id` | `ASKANSWER_TENANT_ID` | 多租户隔离 |
| `db_dialect` | `ASKANSWER_DB_DIALECT` | SQL agent 方言，留空则从连接推断 |
| — | `WLANGGRAPH_POSTGRES_DSN` | SQL agent 默认数据库 |
| `mcp_profile` | `ASKANSWER_MCP_PROFILE` | MCP 自动连接 profile |
| `mcp_all_intents` | `ASKANSWER_MCP_ALL_INTENTS` | MCP 工具对所有意图可见 |
| `server_token` | `ASKANSWER_SERVER_TOKEN` | HTTP Bearer token |
| — | `ASKANSWER_READ_FILE_MAX_BYTES` / `ASKANSWER_WRITE_FILE_MAX_BYTES` / `ASKANSWER_SHELL_OUTPUT_MAX_BYTES` | 读文件 10 MB / 写文件 1 MB / shell 每路输出 64 KB |
| — | `LANGSMITH_API_KEY` / `ASKANSWER_OTEL_EXPORTER` | 可选的可观测导出，关闭时零开销 |

驼峰写法（`tenantId`、`dbPath` 等）同样识别。

### 模型路由与成本

不配 `models` 时与单模型完全一致。配置后：分类 / 评估 / 摘要可用小模型，主回答可设跨厂商回退链；回退会记 `model_fallback` 审计事件，token 归到实际执行的模型；`/status` 显示当前路由，`/usage` 按标签分账（价格表 `askanswer/llm/pricing.py`）。系统提示按「稳定前缀 → 动态尾部」组织以命中 prompt cache，`anthropic:*` 模型自动加 `cache_control`。

离线意图评测：`python evals/run_intent_eval.py`（见 `evals/README.md`）。

---

## 能力

### 意图与工具

内置工具（`tavily_search`、`read_file`、`get_current_time`、`check_weather`、`calculate`、`convert_currency`、`lookup_ip`、`pwd`）对所有意图可见；下表列出每个意图额外拿到的专属工具。

| 意图 | 专属工具 | 说明 |
| --- | --- | --- |
| `chat` | 全部子图工具、shell、`write_file`、MCP 工具 | 通用对话，能力最全 |
| `search` | shell | 联网搜索：Tavily Top 5 + 综合答案 |
| `file_read` | shell、`write_file` | `markitdown` 把 pdf/docx/xlsx/代码转成 Markdown，由主模型分析 |
| `sql` | `sql_query` | 自然语言 → SQL agent 子图；刻意不给 shell |
| `math` | — | 用 `calculate` 安全求值 |
| `research` | `research_brief_loop`、shell | 规划查询 → 搜索 → 综合 → 来源核验 |
| `decision` | `decision_memo_loop`、`helix_spec_loop`、shell | 复用 Helix interview 产出决策备忘 |
| `helix` | `helix_spec_loop` | 规格优先演化（见下） |

**Helix 规格演化**：`interview → seed → execute → evaluate → (重跑 seed | finalize)`，最多 3 代。interview 从 scope / constraints / outputs / verification 四个方向提问并自答（标注 `assumption:`），evaluate 给出覆盖率、0–1 对齐分和缺口。输出 `Goal / Constraints / Acceptance criteria / Artifact / Evaluation / Lineage`。可单独作为 MCP 服务运行：`python -m askanswer.interfaces.mcp_server.helix_server`。

### 澄清

意图 handler 可在首轮回答前提问一次（如文件路径缺失、无 DSN、调研话题过短）。CLI 用箭头菜单加可选自由输入；非交互环境取默认「不改动」选项。

### 确认（HITL）

三类操作共用 `safety/approvals.py` 的四步协议（plan → gate → interrupt → apply）：

| 类别 | 工具 | 用户可选 |
| --- | --- | --- |
| `shell` | `gen_shell_commands_run` | 执行 / 取消 / 重新生成 / 编辑；高风险命令直接拦截 |
| `fs_write` | `write_file` | 查看 diff 后写入 / 取消；敏感路径拦截 |
| `external_api_paid` | 付费外部 API | 查看参数与费用后执行 / 取消；审计脱敏密钥与邮箱 |

---

## 架构

### 主图

单层 7 节点，无子图嵌套：

```
START → route → clarify → reason ─┬─(无工具调用)─→ review ─┬→ END
                                  │                        └→ reason（重试）
                                  ├─(普通工具)───→ act → reason
                                  └─(需确认)─────→ plan_approvals → ask_approval ⟲ → act
```

| 节点 | 职责 |
| --- | --- |
| `route` | 本地分类器（关键词 / 正则，按 handler 优先级）优先，歧义才调 classify 模型；意图参数写入 `slots` |
| `clarify` | handler 可选提问一次，默认直接透传 |
| `reason` | 按意图选系统提示和工具集，调用主模型；唯一向外流式输出 token 的节点 |
| `plan_approvals` | 把需确认的调用规划进 `pending_confirmations` |
| `ask_approval` | 每次只询问一项并把决定写回状态，无副作用，可安全重跑 |
| `act` | 一次性执行普通工具和已批准的操作（`apply` 内二次校验） |
| `review` | 调用 `handler.evaluate` 并检查 token 预算；重试次数按意图（search 2，sql / file_read 1，其他 0） |

节点均由 `make_xxx_node(container)` 生成，依赖来自 `runtime/container.py`。拓扑变更需同步更新 `tests/golden/graph.mmd`。

### 目录

依赖方向：`interfaces → runtime → graph → intents / tools / agents → llm / storage / observability → config / core`

```
askanswer/
├── core/            纯领域类型：Phase / Risk / capability
├── config/          Settings、loader（唯一读环境变量处）、分层 settings.json、路径
├── llm/             模型代理（/model 热切换）、角色路由与回退、历史预算、价格表
├── graph/           state、ContextSchema、edges、builder
│   └── nodes/       route / clarify / reason / approvals / act / review
├── intents/         chat / search / file_read / sql / math / helix / research / decision
├── tools/           ToolRegistry（风险、能力、确认声明校验）+ 内置工具
├── agents/          sql / helix / research / decision 子图，注册为工具
├── mcp/             多 server 生命周期、工具包装、mcp.json
├── safety/          确认协议
├── storage/         db（连接 + WAL + 迁移）、models、repos/{threads,audit,labels}、Store 门面
├── observability/   事件总线、审计订阅者、telemetry（LangSmith / OTEL）
├── runtime/         Container（组装根）、runner（UI 无关事件流）、时间旅行
└── interfaces/
    ├── cli/         REPL 与斜线命令
    ├── http/        HTTP/SSE 服务
    └── mcp_server/  Helix MCP 服务
```

### 持久化与可观测

- **状态库**：一个 SQLite 文件（默认 `~/.askanswer/state.db`）同时存 LangGraph checkpoint 和自有表 `thread_meta` / `audit_event` / `checkpoint_label`，schema 自动迁移，按 `tenant_id` 隔离
- **存储分层**：`storage/db.py` → `storage/repos/*`（每表一个 repo）→ `storage/store.py`（`get_store()`）；删除会话走 `SqliteSaver.delete_thread`
- **事件总线**：业务代码只调用 `log_event` 发布事件；审计（按轮缓冲、批量落库）和 telemetry 都是订阅者，订阅者出错不影响主流程
- **时间旅行**：`/checkpoints` `/undo` `/jump` `/fork`；有挂起确认时拒绝回退

---

## 开发

```bash
pip install -e ".[dev]"
pytest -q                        # 无需 API key；conftest 隔离环境变量与临时 DB
ruff check askanswer tests       # 规则 E / F / I
python -m askanswer --graph -    # 校验图可编译
```

CI（`.github/workflows/ci.yml`）在 push / PR 到 `master` 时对 Python 3.10 与 3.12 运行 ruff + pytest。

**扩展**：新意图实现 `IntentHandler` 并注册到 `IntentRegistry`；新工具注册到 `ToolRegistry` 并声明 tags、风险与确认类别。细节见 `plan-docs/mem/architecture-extensibility.md`。

**文档**：架构笔记 `plan-docs/mem/` · 重构状态 `docs/design/ARCH_REFACTOR_STATUS.md` · 变更日志 `CHANGELOG.md` · 待办 `TODO.md`
