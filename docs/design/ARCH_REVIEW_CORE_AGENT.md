# AskAnswer（`core-agent`）架构评审与结构重构方案（修订版）

- **目标版本**：`askanswer` v0.1.0（分支 `core-agent` @ commit `2d22c7f`）
- **代码规模**：50 个 Python 模块（8,197 LOC）；包根目录 20 个文件（5,330 LOC）；10 个测试文件（1,103 LOC，111 个测试）
- **范围**：抽取后（post-extraction）的架构评估与包结构重构计划
- **修订说明**：本文是原评审文档（英文版 `ORIGINAL_REVIEW.md`）的中文修订版。所有结论已逐条对照
  `2d22c7f` 的代码核对；与代码不符或实施中证明不妥之处，统一记录在 [§8 核对结论 / 勘误](#8-核对结论--勘误)。
  实施进度见文末 [§9 状态表](#9-实施状态表)。代码标识符保持英文原样。

---

## 1. 摘要

Commit `2d22c7f`（*"refactor: extract embeddable agent core, drop CLI and HTTP server"*）删除了近万行应用 / UI 脚手架
（`cli/`、`server.py`、`web/`、`wire.py` 等），留下一个快速、测试完善、可嵌入的 LangGraph agent 库。
111 个单元测试约 1.7 秒跑完，导入时不需要网络或 API key。

但由于该库是从一个更大的单体应用里"切"出来的，**其内部模块拓扑仍然是"应用布局"而非"可嵌入 SDK 布局"**：

1. **顶层命名空间膨胀**：包根目录平铺 20 个文件（5,330 LOC，约占 65%），混杂了图节点、SQLite DDL/迁移、
   shell 沙箱、MCP 事件循环、LLM 代理路由等。
2. **嵌入隐患**：`import askanswer` 会在导入期执行 `settings.bootstrap_environ()`（修改宿主进程的 `os.environ`）；
   `run_leg()` 即使宿主注入了 `InMemorySaver`，仍会把审计 / thread 元数据写入全局 SQLite 单例。
3. **上帝模块与职责割裂**：`persistence.py`（872 LOC）、`tools.py`（589）、`mcp.py`（557）、`registry.py`（409）
   各自捆绑了多个正交关注点。
4. **`2d22c7f` 的附带损伤**：抽取时删除了 CI workflow（`.github/workflows/ci.yml`）、配置模板
   （`.env.example`、`settings.example.json`），并让被跟踪的 `CLAUDE.md` 指向不存在 / 被 gitignore 的文件。

---

## 2. 架构优势（重构必须保持的不变量）

1. **意图无关的父图拓扑**：父图恒为 `START → understand → answer → sorcery → (END | answer)`，不按 intent 分叉。
   intent 差异由 `IntentHandler` 协议参数化（分类、可选澄清、prompt 提示、工具包、`sorcery` 重试评估）。
2. **四步持久化 HITL 确认协议**：`plan() → gate() → interrupt_payload() → apply()` 分布在
   `_confirm_plan_node` 与 `_tools_node`；在 `interrupt()` **之前**把要执行的具体动作写入
   `state["pending_confirmations"]`，并在 `apply()` 中对人工编辑后的内容重新过安全闸门。
3. **状态外可观测性**：审计缓冲（`audit`）与 telemetry span 位于 `contextvars`，不进 `SearchState`，checkpoint 保持小而干净。
4. **零成本禁用路径与惰性初始化**：导入不需要 API key、不打开 SQLite / 网络；`ASKANSWER_CONTEXT_MAX_TOKENS`
   未设置时零 token 估算；telemetry 关闭时不 import `langsmith` / `opentelemetry`。
5. **对 prompt cache 友好的上下文组装**：`answering.py` 先放稳定指令与工具清单、后放动态上下文（Anthropic 下带
   `cache_control` 断点）；`context.py` 以原子块裁剪历史，`AIMessage` 的 tool call 与 `ToolMessage` 永不拆散。

---

## 3. 架构问题与技术债

### 3.1 可嵌入性与宿主隔离

| 问题 | 位置 | 对宿主的影响 |
| :--- | :--- | :--- |
| **导入期修改环境变量** | `load.py` 顶层调用 `settings.bootstrap_environ()` | 仅 `import askanswer` 就会从 `cwd` 向上查找并把 `.askanswer/settings*.json` 与 `.env` 写进 `os.environ`，可能悄悄改变宿主行为，或在只读 / 容器环境中失败。 |
| **导入期构造外部客户端**（勘误 f） | `load.py`：`tavily_client = TavilyClient(api_key=os.getenv(...))`、`openweather_api_key = os.getenv(...)` | 即便把 bootstrap 改为惰性，这两个值仍在导入期固化，settings.json 里的 key 永远读不到；必须同时改为惰性 getter。 |
| **`run_leg()` 持久化泄漏** | `runner.py`（`flush_pending` + `_upsert_thread_meta` → `get_persistence()`） | `create_search_assistant(checkpointer=InMemorySaver())` 构图时不碰 SQLite，但 `run_leg()` 仍会初始化 `~/.askanswer/state.db`。 |
| **导入期捕获 env 开关** | `registry.py:49`（`ASKANSWER_MCP_ALL_INTENTS`） | 模块级常量，导入后再改 env 不生效。 |
| **遗留 DSN 变量名** | `runner.py`、`sqlagent/sql_interact.py`，以及 `intents/sql.py` 的澄清提示文案 | 读取 `WLANGGRAPH_POSTGRES_DSN`（早期仓库的遗留名），而非 `ASKANSWER_DB_DSN`。 |

### 3.2 单体模块与职责纠缠

1. **`persistence.py`（872 LOC）**：SQLite PRAGMA / 锁、schema 迁移 DDL（v1–v4）、`SqliteSaver` 装配、
   `thread_meta` COALESCE upsert、`audit_event` 批量写入、`checkpoint_label` CRUD、跨租户鉴权、用量汇总 SQL 全在一个文件。
   另外 `audit.py` 的 `LLMUsageCallback` 在内存中收集了 `cached_tokens`，但落库时被丢弃（注释写着"等 schema v5"）。
2. **`tools.py`（589 LOC）+ `confirmations.py`（328 LOC）**：`tools.py` 把 8 个简单内置工具与安全机制（敏感路径正则、
   `describe_write` diff、LLM 生成 shell 命令、`_DANGEROUS_PATTERNS`）混在一起；而 `write_file` /
   `gen_shell_commands_run` 的真实执行在 `confirmations.py` 中，`tools.py` 里只剩兜底空壳。
3. **`registry.py`（409 LOC）+ `mcp.py`（557 LOC）**：`registry.py` 本应是纯粹的 `ToolRegistry`，却还硬编码了所有内置 /
   子 agent 的 seed，以及约 120 行 MCP JSON-Schema → Pydantic 的动态模型转换。

### 3.3 子 agent 分散、耦合与废弃依赖

- **顶层分散**：`decision/`、`helix/`、`research/`、`sqlagent/` 与基础设施包（`telemetry/`、`intents/`）平级。
- **跨子 agent 耦合**：`decision/agent.py` 直接 `from ..helix.nodes import interview_node`。苏格拉底访谈是子 agent 共享原语，不是 Helix 的实现细节。
- **`langchain-community` 停止维护警告**：`sqlagent/sql_interact.py` 从 `langchain_community` 导入
  `SQLDatabase` 与 `SQLDatabaseToolkit`，触发测试套件里唯一的 `DeprecationWarning`。注意（勘误 d）：sqlagent
  使用的是 toolkit 的**全套**工具，`sql_node.py` / `sql_agent.py` 按名字引用 `sql_db_list_tables` /
  `sql_db_schema` / `sql_db_query`（`sql_db_query_checker` 内部调用 LLM），替换实现必须保持这些工具名与行为。

### 3.4 HITL 与运行时边界情况

- **`PaidApiConfirmation` 绕过 `ToolNode`**：`confirmations.py` 中 `PaidApiConfirmation.apply()` 直接
  `descriptor.tool.invoke(args)`，不经 `ToolNode`、也不注入 `ToolRuntime[ContextSchema]`；任何依赖 `ContextSchema`
  （租户、每次调用凭据）的付费工具在 resume 后都会失败。（同类低风险点见勘误 g：`research/nodes.py` 也直接
  `descriptor.tool.invoke`。）
- **OTEL span 扁平**（勘误 e）：`telemetry/__init__.py` 已经计算 `parent_id`，LangSmith exporter 也会嵌套；
  只有 `telemetry/otel.py` 忽略 `parent_id`，导致 OTEL 后端看到的是扁平 span。

### 3.5 抽取后的遗留（`2d22c7f`）

- **0.1.0 之前的兼容垫片**（已全部核实存在）：
  - `SearchState.pending_shell`（已被 `pending_confirmations` 取代）
  - `BUNDLE_*` 常量与 `ToolDescriptor.bundles` / `requires_confirmation` 别名属性
  - `tools.tools` 模块级列表与 `tools.tools_by_name` 字典
  - `MCPClientManager.connect()` 别名，以及 `_unused` 占位 schema hack（勘误 k：该 hack 实际已失效）
- **缺失的仓库 / CI 资产**：
  - `.github/workflows/ci.yml` 被删除，且 `.github/workflows/` **目录本身不存在**（勘误 a）。
  - `.env.example`、`settings.example.json` 被删除。
  - `CLAUDE.md` 被 git 跟踪，却链接到 `.claude/mem/commands.md`、`.claude/mem/architecture.md`（不存在，且 `.claude/` 被 gitignore）。
- **测试盲区**：`persistence`、`settings`、`confirmations`、`registry` 覆盖良好，但 `context.py`、`routing.py`、
  `audit.py`、`clarify.py` 以及四个子 agent（`helix/`、`decision/`、`research/`、`sqlagent/`）**没有任何专属测试文件**。

---

## 4. 目标包结构

将 50 个模块重组为 **8 个内聚的领域子包**（`config`、`llm`、`core`、`intents`、`tools`、`subagents`、`storage`、
`observability`；原文写作"6 个"，见勘误 b），同时保证 `askanswer/__init__.py` 的公开导出与原来**完全一致**。

```text
AskAnswer/
├── pyproject.toml            # 去掉 langchain-community；显式声明 sqlalchemy>=2.0；补 readme/urls 元数据
├── README.md                 # 双语概述 + 嵌入指南（更新为新布局）
├── CHANGELOG.md              # 恢复并记录本次重构（含不兼容项）
├── .env.example              # 恢复（去掉 VOYAGE_*，新增 ASKANSWER_DB_DSN）
├── settings.example.json     # 恢复
├── CLAUDE.md                 # 自包含的 agent 指南（无失效链接）
├── .github/workflows/ci.yml  # 恢复：ruff + pytest，Python 3.10 / 3.12
├── docs/design/ARCH_REVIEW_CORE_AGENT.md
├── askanswer/
│   ├── __init__.py           # 公开门面（导出不变）
│   ├── config/               # settings.py（分层配置 + 惰性 bootstrap）、schema.py（ContextSchema）
│   ├── llm/                  # proxy.py（_ModelProxy / set_model）、routing.py（ROLE_* / RoutedModel）
│   ├── core/                 # state, graph, nodes, runner, context, clarify, confirmations
│   │   └── react/            # __init__（build_react_subgraph）、answering.py、dispatch.py
│   ├── intents/              # 保持原结构
│   ├── tools/                # registry, builtin, filesystem, shell, mcp_manager, mcp_adapter
│   ├── subagents/            # common/interview.py, decision/, helix/, research/, sql/
│   ├── storage/              # manager.py（含 NullPersistence）, migrations.py, threads.py, audit_store.py
│   └── observability/        # audit, redaction, telemetry, langsmith, otel
└── tests/                    # core/ config/ llm/ intents/ tools/ subagents/ storage/ observability/
```

> 关于 `LICENSE`：原文目录树列出"新增 LICENSE"，但仓库作者未声明许可证，本次重构**不擅自选择许可证**（勘误 l）。

---

## 5. 详细重构规格

### 5.1 拆分 `persistence.py` 并修复 `InMemorySaver` 泄漏

1. 拆成 `askanswer/storage/`：
   - `migrations.py`：`askanswer_schema` 版本迁移 v1–v5；**v5** 执行
     `ALTER TABLE audit_event ADD COLUMN cached_tokens INTEGER DEFAULT 0`，使 `cached_tokens` 落库。
   - `threads.py`：`upsert_meta`、`list_threads`、`get_meta`、`find_by_prefix`、`set_title`、`delete_thread`、`checkpoint_label`。
   - `audit_store.py`：`log_audit_event(s)`、`list_audit_events`、`usage_summary`（新增 `cached_tokens` 汇总）、`import_audit_events`。
   - `manager.py`：连接生命周期、WAL、`SqliteSaver` 装配、单例；以及**新增**的 `NullPersistence`（勘误 c：它目前并不存在，是新代码而非"保留"）。
2. `run_leg()` 尊重非 SQLite checkpointer：新增 `persistence=` 关键字参数。解析顺序：显式传入 →
   `app.checkpointer` 恰好是某个 `PersistenceManager` 的 saver 时用该 manager → 否则 `NullPersistence`。
   审计 flush 与 thread_meta 写入都走解析出的后端；纯内存运行不会创建任何 SQLite 文件。

### 5.2 拆分 `tools.py` 与 `registry.py`

1. 按风险域拆分：`tools/builtin.py`（只读工具 + 惰性 `get_tavily_client()` / OpenWeather key）、`tools/filesystem.py`
   （`read_file` / `write_file` / 敏感路径正则 / `validate_write_path` / `describe_write`）、`tools/shell.py`
   （`gen_shell_commands_run` / `execute_shell_command` / `_DANGEROUS_PATTERNS`）。
2. MCP schema 转换（`_jsonschema_to_pydantic` / `_wrap_mcp_tool`）移入 `tools/mcp_adapter.py`；`ASKANSWER_MCP_ALL_INTENTS`
   在 `refresh_mcp()` 中动态读取。

### 5.3 子 agent 合并与去除 `langchain-community`

1. 共享访谈节点 `subagents/common/interview.py`，`helix` 与 `decision` 都从这里导入。
2. `subagents/sql/` 用原生 `sqlalchemy`（`create_engine` + `inspect`）实现与 `SQLDatabaseToolkit` **同名同语义**的四个工具
   （`sql_db_list_tables`、`sql_db_schema`、`sql_db_query`、`sql_db_query_checker`，后者经模型代理调用 LLM）以及
   `get_schema` / `find_slow_sql`。工作量明显超过原文估计的 ~60 LOC（勘误 d）；`sqlalchemy>=2.0` 必须显式声明。
3. DSN 环境变量：`ASKANSWER_DB_DSN` 为主；`WLANGGRAPH_POSTGRES_DSN` 仅作已废弃回退（读取时发出 `DeprecationWarning`）。

### 5.4 把 settings bootstrap 从导入期解耦

- 删除 `load.py` 顶层的 `bootstrap_environ()` 调用。
- 在首次 `create_search_assistant()`、模型首次使用（`_ModelProxy` 构造 backend / 解析默认模型标签、`model_for()`）、
  `get_persistence()`、`runtime_context_from_env()` 时惰性调用 `ensure_bootstrapped()`。
- 宿主可调用 `bootstrap_environ(enabled=False)` 完全退出文件配置加载（只用进程 env 与 `ContextSchema`）。
- 连带要求（勘误 f/j）：所有"导入期读 env"的值都必须改为调用期读取（Tavily client、OpenWeather key、默认模型标签、
  读写文件 / shell 输出大小上限）。

### 5.5 `PaidApiConfirmation` 走 `ToolNode`

`apply()` 改为用单次调用的 `ToolNode([descriptor.tool])` 执行（在 react 子图节点内运行，自动继承父图的
`Runtime[ContextSchema]`），从而 `ToolRuntime` 注入在 resume 后依然有效。

### 5.6 清理 0.1.0 之前的兼容垫片

- 移除 `SearchState.pending_shell` 与 dispatch 中的读取（勘误 h：这会放弃对升级前挂起 checkpoint 的 resume 兼容，
  在 0.1 之前可以接受，已写入 CHANGELOG）。
- 移除 `BUNDLE_*`、`.bundles`、`.requires_confirmation`、`tools`、`tools_by_name`、`MCPClientManager.connect()`、`_unused` hack。

---

## 6. 仓库卫生与文档

1. 恢复 `.github/workflows/ci.yml`（`ruff check askanswer tests` + `pytest`，Python 3.10 / 3.12）。
2. 恢复 `.env.example`、`settings.example.json`（去掉无引用的 `VOYAGE_*`，新增 `ASKANSWER_DB_DSN`）。
3. `CLAUDE.md` 改为自包含（内联命令、架构与配置不变量）。
4. 公开 API 模块（`__init__.py`、`runner.py`、`graph.py`、`schema.py`、`registry.py`、`intents/base.py` 等）的模块头与公开
   docstring 统一为英文（或中英双语）。

---

## 7. 分阶段迁移路线

| 阶段 | 范围 | 验证门槛 |
| :--- | :--- | :--- |
| **Phase 1：存储与可观测性** | `storage/` 拆分 + schema v5；`audit`/`telemetry` → `observability/`（含 `redaction`、OTEL 嵌套）；`run_leg()` 的 `NullPersistence` | `pytest tests/storage tests/observability`；`InMemorySaver` + `run_leg()` 不创建 SQLite 文件 |
| **Phase 2：工具与 MCP** | `tools/` 拆分；移除 `BUNDLE_*` / `tools_by_name` 等垫片；动态 MCP 开关 | `pytest tests/tools`；ruff 干净 |
| **Phase 3：子 agent 与 SQLAlchemy** | `subagents/` 迁移；`common/interview.py`；原生 sqlalchemy；DSN 变量 | `pytest` **0 warning**；新增 `tests/subagents/` |
| **Phase 4：core / config / llm** | `config/`、`llm/`（惰性 bootstrap）、`core/`；`PaidApiConfirmation` 走 `ToolNode` | 全量测试 + 导入冒烟（`os.environ` 不变、无 `state.db`） |
| **Phase 5：CI、文档、补测试** | CI / 模板 / `CLAUDE.md` / README / CHANGELOG / pyproject；补 context、clarify、routing、audit 测试 | ~145+ 测试通过 |

每个 commit 之后：全量 `pytest` 通过，`ruff check askanswer tests` 干净。

---

## 8. 核对结论 / 勘误

原文中以下事实**已核实属实**：模块 / LOC 统计（50 模块 / 8,197 LOC；根目录 20 文件 / 5,330 LOC；10 个测试文件 / 1,103 LOC；
111 个测试）；`load.py` 导入期 `bootstrap_environ`；`run_leg` 的 `flush_pending` + `_upsert_thread_meta` 无视 checkpointer 命中
`get_persistence()`（SQLite）；`registry.py:49` 的模块级 `ASKANSWER_MCP_ALL_INTENTS`；`runner.py` 读取
`WLANGGRAPH_POSTGRES_DSN`（`sqlagent/sql_interact.py` 与 `intents/sql.py` 的提示文案亦然）；`cached_tokens` 落库时被丢弃；
`decision` 从 `helix.nodes` 导入 `interview_node`；`PaidApiConfirmation` 直接调用 `descriptor.tool.invoke`；所有遗留垫片均存在；
`CLAUDE.md` 链接到被 gitignore 的 `.claude/mem`；`ci.yml` / `.env.example` / `settings.example.json` 均在 `2d22c7f` 中被删除。

需要更正或补充的内容：

- **(a)** `.github/workflows/` 并非"空目录"——它**根本不存在**（git 不跟踪空目录），删除 `ci.yml` 后整个路径都消失了。
- **(b)** §4 写"6 个内聚子包"，但列出了 8 个（`config`、`llm`、`core`、`intents`、`tools`、`subagents`、`storage`、`observability`）。正确数字是 8。
- **(c)** `NullPersistence` 目前**不存在**；§5.1 的 "retain" 措辞有误，它是本次新增代码。
- **(d)** sqlagent 使用 `SQLDatabaseToolkit` 的**全套**工具：`sql_db_list_tables` / `sql_db_schema` / `sql_db_query` /
  `sql_db_query_checker`（checker 调用 LLM），且 `sql_node.py`、`sql_agent.py` 按名字引用它们。替换必须保持这些名字与行为
  （含 `sql_db_schema` 的 `CREATE TABLE` + 3 行样例格式、`sql_db_query` 的 `Error: ...` 错误回传与 `str(list[tuple])` 结果格式），
  实际代码量远超 ~60 LOC；`sqlalchemy` 必须显式声明为依赖。
- **(e)** OTEL：`telemetry/__init__.py` 已经计算 `parent_id`，LangSmith exporter 也已嵌套；只有 `otel.py` 忽略 `parent_id`。
  修复方式：在 `otel.py` 中用父 span 的上下文（`trace.set_span_in_context(parent)`）启动子 span。
- **(f)** 额外发现：`load.py` 在导入期构造 `tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))` 并读取
  `OPENWEATHER_API_KEY`。若只把 bootstrap 改为惰性而不同时把它们改为惰性，settings.json 中配置的这两个 key 将永远读不到。
  已改为 `tools/builtin.py` 中的惰性 getter。
- **(g)** `research/nodes.py` 同样直接 `descriptor.tool.invoke` 调用 `tavily_search`（低风险：该工具不依赖 `ToolRuntime`），仅记录，暂不改动。
- **(h)** 移除 `pending_shell` 会放弃对"升级前挂起的 shell 确认 checkpoint"的 resume 兼容。0.1 之前可以接受，已记录在 CHANGELOG。
- **(i)** 额外发现：`telemetry.init_telemetry()` 在整个代码库中**从未被调用**，即使设置了 `LANGSMITH_API_KEY` /
  `ASKANSWER_OTEL_EXPORTER`，exporter 也永远不会装配。修复：`open_span()` / `llm_callback()` 首次调用时惰性 `init_telemetry()`
  （仍保持"未启用即零开销、不 import SDK"）。
- **(j)** 额外发现：惰性 bootstrap 还要求其它导入期 env 读取同步改为调用期读取，否则 settings.json 中的值会被忽略：
  `load._initial_model()`（`ASKANSWER_DEFAULT_MODEL`，导入期固化默认模型）、`tools.py` 的
  `ASKANSWER_READ_FILE_MAX_BYTES` / `ASKANSWER_WRITE_FILE_MAX_BYTES` / `ASKANSWER_SHELL_OUTPUT_MAX_BYTES`。均已改为调用期读取
  （模块级常量保留为可选的测试覆盖钩子）。
- **(k)** 额外发现：`_unused` 占位 hack 在 Pydantic v2 下**本来就是坏的**——`create_model` 拒绝下划线开头的字段名并抛
  `PydanticUserError`，`_jsonschema_to_pydantic` 因而返回 `None`，**零参数 MCP 工具被静默跳过**。删除 hack 后用无字段模型即可正常包装（已加测试）。
- **(l)** 原文目录树中的 `LICENSE`：作者未声明许可证，重构不应代为选择，本次不添加（留待作者决定）。
- **(n)** 额外发现：`bootstrap_environ()` 调用 `load_dotenv()` 时未传路径，python-dotenv 默认从**调用方源文件所在目录**
  向上查找 `.env`——对库代码而言是 `site-packages/askanswer`，而不是宿主的工作目录（文档声称"从 cwd 向上找"）。
  在仓库内开发时恰好一致，安装成依赖后则读不到宿主的 `.env`。已改为 `find_dotenv(usecwd=True)`。
- **(m)** 原文 Phase 4 冒烟命令 `env -u OPENAI_API_KEY python -c "import askanswer"` 不足以证明"导入无副作用"。改为在干净环境中
  比较导入前后的 `os.environ` 并断言 `$HOME/.askanswer/state.db` 不存在（见 §9 与 CHANGELOG）。

---

## 9. 实施状态表

| 阶段 / 项 | 状态 | 说明 |
| :--- | :--- | :--- |
| Step 0：修订评审文档 | ✅ 完成 | 本文件 |
| Phase 1：storage 拆分 + schema v5 + NullPersistence | ✅ 完成 | `storage/{manager,migrations,threads,audit_store}.py`；v5 `cached_tokens`；`run_leg(persistence=)` + `NullPersistence`，`InMemorySaver` 运行不建库（有测试） |
| Phase 1：observability（audit / redaction / telemetry / OTEL 嵌套） | ✅ 完成 | 另修复勘误 (i)：exporter 首次使用时惰性装配 |
| Phase 2：tools 拆分 + 去垫片 + 动态 MCP 开关 | ✅ 完成 | `tools/{registry,builtin,filesystem,shell,mcp_manager,mcp_adapter}.py`；Tavily / OpenWeather / 大小上限改为调用期读取（勘误 f/j）；零参 MCP 工具修复（勘误 k） |
| Phase 3：subagents + 原生 SQLAlchemy + DSN 变量 | ✅ 完成 | `subagents/{common,helix,decision,research,sql}`；`sql/database.py` 原生实现，四个工具与原 toolkit 并排比对输出一致；测试 0 warning；访谈节点新增失败兜底 |
| Phase 4：config / llm / 惰性 bootstrap / core | ✅ 完成 | `config/`、`llm/`、`core/`（`react/{__init__,answering,dispatch}`）；`ensure_bootstrapped()` + `bootstrap_environ(enabled=False)`；去掉 `pending_shell`；`PaidApiConfirmation` 经 `ToolNode` 执行（端到端 HITL 测试验证租户上下文注入）；导入冒烟：`os.environ` 不变、无 `state.db` |
| Phase 5：CI / 模板 / CLAUDE.md / README / CHANGELOG / 补测试 | ✅ 完成 | CI（ruff + `pytest -W error` + 导入冒烟，3.10/3.12）；`.env.example` / `settings.example.json`；自包含 `CLAUDE.md`；README / pyproject / CHANGELOG；新增 context / clarify / routing / audit 测试 |

**最终结果**：测试 111 → 272（0 warning，`pytest -W error` 通过，约 4 秒）；`ruff check askanswer tests` 干净；
`askanswer/__init__.py` 导出不变；`import askanswer` 不修改 `os.environ`、不创建 `state.db`。
未做：`LICENSE`（勘误 l）；`research/nodes.py` 的直接 `tool.invoke`（勘误 g，低风险，仅记录）。
