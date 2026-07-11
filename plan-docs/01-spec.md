# AskAnswer 系统规格（简版）

> 描述系统「是什么、必须保持什么」。实现细节看 `.claude/mem/architecture-*.md`；本文只锁定对外行为与不变量，是 review 与测试的依据。

## 1. 产品定位

CLI 优先的 LangGraph 智能助手：按意图把用户请求分流到 本地文件读取 / 联网搜索 / SQL 查询 / 规格演化(Helix) / 研究简报 / 决策备忘 / 直接对话，由 LLM 经受治理的工具集完成任务，带自评重试、HITL 确认、审计计费、多租户持久化与 MCP 双向互操作。

## 2. 系统组成

### 2.1 编排核心

- 父图拓扑**固定**：`START → understand → answer(ReAct 子图) → sorcery → (END | answer)`。
- `SearchState` 只承载消息与控制字段（见 `state.py`）；节点一律返回 partial dict。
- 重试通过 `retry_directive` 结构化指令，不伪造 HumanMessage。

### 2.2 扩展机制

- **Intent**：`IntentHandler` 协议 + `IntentRegistry`（`intents/__init__.py` 注册 8 个：file_read, sql, helix, decision, math, research, search, chat）。新意图 = 新 handler 文件 + 注册，不改父图。
- **Tool**：一切工具经 `ToolRegistry`（tag 过滤 + `confirmation_class` 元数据）。子图作为工具暴露：`sql_interact`、`helix_spec_loop`、`research_brief_loop`、`decision_memo_loop`。
- **MCP**：客户端 `MCPClientManager`（stdio/url、健康探测、`~/.askanswer/mcp.json` profile 自动重连）；服务端 `helix_mcp.py`（stdio 暴露 Helix，禁止 import `graph.py`）。

### 2.3 治理

- **HITL 确认**：`confirmations.py` 统一协议（plan → gate → interrupt → apply），三类：`shell`、`fs_write`（敏感路径正则 + 1 MB 上限 + diff 预览）、`external_api_paid`（费用提示 + 参数脱敏）。挂起态存 `pending_confirmations`。
- **审计与计费**：`audit.py` + `pricing.py` 落 SQLite，`/audit` `/usage` 查询；事件带 `tenant_id`。
- **多租户**：`ASKANSWER_TENANT_ID` → `ContextSchema.tenant_id`；persistence 读写按 tenant 过滤（None = 不限，向后兼容）；SQL 连接缓存按 `(tenant_id, dsn)` 键控。

### 2.4 持久化（SQLite，schema **v4**）

- `~/.askanswer/state.db`（`ASKANSWER_DB_PATH` 可覆盖）；LangGraph `SqliteSaver` 管 checkpoint，自建表：`thread_meta`（含 `tenant_id`）、`audit_event`（含 `tenant_id`）、`checkpoint_label`。
- 迁移链 v2→v3（tenant 列+索引）→v4（`checkpoint_label` 表）必须对旧库无损。
- 时间旅行：`/checkpoints` `/undo [n] [--label]` `/jump` `/fork`；有挂起确认时拒绝 rewind。

### 2.5 可观测性（opt-in，关闭时零开销）

`telemetry/`：LangSmith（`LANGSMITH_API_KEY`）与 OpenTelemetry（`ASKANSWER_OTEL_EXPORTER`）导出器；contextvar span 栈；不写 `SearchState`，不在热路径顶层 import。

## 3. 不变量（禁止破坏）

1. 父图拓扑固定，禁止按 intent 分叉。
2. 工具必须经 `ToolRegistry`，禁止直连 `bind_tools`。
3. `graph.py` 顶层禁止 import persistence/audit；`get_persistence()` 只在 CLI 入口与 saver 注入处触发。
4. `SearchState` 不塞 telemetry/metrics 大对象。
5. SQL agent 保持只读拦截。
6. 节点返回 partial dict，永不返回完整 state。
7. 副作用工具的执行体必须在确认 resume 之后运行，工具本体不得先落盘/先调用。
8. 未设 `ASKANSWER_TENANT_ID` 时行为与单租户完全一致（非回归）。

## 4. 质量门槛（对所有新写/触碰的代码强制）

- 函数 ≤ 50 行；文件 ≤ 300 行；嵌套 ≤ 3 层；位置参数 ≤ 3；圈复杂度 ≤ 10；魔法数字必须提取常量。
- 存量超限文件（`cli.py` 2466、`persistence.py` 875、`tools.py` 581、`mcp.py` 547）在 G3-C1 前豁免「文件行数」一项，但对其新增/修改的**函数**仍执行函数级限制。
- 提交格式 `<type>: <description>`；提交前过安全清单（无硬编码密钥、输入校验、参数化 SQL、错误信息不泄密）。

## 5. 验证基线

- 无 API key 也能跑的冒烟矩阵：包导入、`askanswer --graph`、空库迁移到 v4、v2 旧库升级、tenant 过滤、mcp_profile 读写、telemetry 关闭态零副作用、`helix_mcp` 不触发 persistence。
- G2 起以 `pytest` 固化上述矩阵；`ruff check askanswer` 为 lint 基线。
