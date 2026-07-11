# AskAnswer 可能性空间探索

## 1. 现状一句话总结

AskAnswer 已是一个 CLI 优先的 LangGraph agent：主图固定为 `understand -> answer(ReAct 子图) -> sorcery`，通过 `IntentHandler`、`ToolRegistry`、SQL/Helix 子图工具、SQLite 持久化、MCP 与审计命令形成了可扩展但仍偏单机场景的产品骨架。

补充坐标：`enterprise-persistence-plan.md` 与源码显示阶段 A/B/C 已落地，`/undo` 不是待实现项；它更适合作为后续加固对象。README 部分描述仍有旧版“合成 HumanMessage”痕迹，权威实现已是 `retry_directive`。

## 2. 四条主线方向卡片

### 线 1：新 Intent / 新 Subgraph-as-Tool

| 方向 | 触达模块 | 抽象改动 | 风险与不变量影响 | 验证方式 | 工作量 |
|---|---|---|---|---|---|
| `code_review` 代码审阅子图 | `askanswer/intents/code_review.py`、`askanswer/code_review/`、`registry.py` | 新 `CodeReviewHandler`；`prompt_hint`: “先读取目标文件/目录摘要，再按风险排序输出 review”；`bundle_tags={"code_review","file_read","search"}`；`max_retries=1`；子图 `collect_files -> analyze -> rank_findings -> finalize` | 安全。父图不变，工具仍走 registry；需避免直接递归读大目录 | 用小型 repo 文件触发，检查是否调用 `read_file`，sorcery 能对“无行号/无风险排序”重试 | M |
| `data_profile` 数据剖析子图 | `intents/data_profile.py`、`askanswer/dataprofile/`、`sqlagent/` | `prompt_hint`: “先采样 schema/字段分布，再给质量报告”；`bundle_tags={"data_profile","sql","file_read"}`；`max_retries=1`；子图复用 SQL agent 或读 CSV 后做 `profile -> anomalies -> recommendations` | 基本安全。要继承 SQL 只读约束，不新增直连 DB 路径 | SQLite/Postgres 测试表、CSV 文件各跑一次；确认不执行 DML | M |
| `research_brief` 研究简报子图 | `intents/research.py`、`askanswer/research/`、`registry.py` | `prompt_hint`: “先搜索多源，再交叉核验、列引用”；`bundle_tags={"research","search"}`；`max_retries=2`；子图 `plan_queries -> search -> synthesize -> source_check` | 安全。主图不变；要避免在 handler 里硬绑 Tavily，必须通过工具调用 | 问“最近某政策/产品变化”，确认多次 `tavily_search` 和带来源摘要 | M |
| `decision_memo` 决策备忘子图 | `intents/decision.py`、`askanswer/decision/`、可复用 `helix/` | `prompt_hint`: “澄清目标/约束/方案，再输出 tradeoff memo”；`bundle_tags={"decision","helix","search"}`；`max_retries=0`；可让 Helix interview 作为前置澄清 | 安全。Helix 自评估循环已覆盖，不应再让父 sorcery 重试 | 非 TTY 下用 default_answer；TTY 下验证可交互澄清 | S/M |

### 线 2：Tool / 生态扩展

| 方向 | 触达模块 | 抽象改动 | 风险与不变量影响 | 验证方式 | 工作量 |
|---|---|---|---|---|---|
| `fs_write` 确认类落地 | `registry.py`、`_react_internals.py`、`tools.py`、`cli.py` | 新工具 `write_file_patch` 或 `propose_file_write`；`confirmation_class="fs_write"`；执行层展示 path、diff、大小、敏感路径检查 | 安全但高风险。必须通过 ToolRegistry；不能让工具直接写文件绕过确认 | 生成补丁，确认前不落盘；拒绝 `.env`、SSH key、超大写入 | M |
| `external_api_paid` 确认类 | `registry.py`、`_react_internals.py`、`cli.py`、`mcp.py` | 对付费/副作用 API 工具加 cost/risk metadata；CLI 展示预计费用/目标 host/参数摘要 | 安全。已有枚举口子；注意审计参数脱敏 | 注册一个 mock paid tool，确认批准/拒绝路径和 audit | M |
| MCP 健康与 profile 连接 | `mcp.py`、`registry.py`、`cli.py`、可选 config 文件 | 给 `_ServerEntry` 加 `last_checked/status/error`；`/mcp health`；启动时从 profile 连接 stdio/url | 不破坏不变量。MCP 仍由 manager 管，registry 只 refresh | 启停一个 stdio server，确认断开后 registry 移除工具 | M |
| Helix 作为 MCP server 暴露 | 新 `askanswer/mcp_server.py` 或 `askanswer/helix_mcp.py`、`helix/agent.py` | 将 `helix_spec_loop(topic)` 包成 MCP tool，给其它 agent 调用；复用 `run_helix_agent` | 安全。不能让 MCP server import `graph.py` 顶层触发 persistence；只暴露 Helix 子图 | 外部 MCP client list/call tool；非 TTY 回退 default_answer | M |

### 线 3：运行时与企业化

| 方向 | 触达模块 | 抽象改动 | 风险与不变量影响 | 验证方式 | 工作量 |
|---|---|---|---|---|---|
| `/undo` 加固与 schema v3 | `timetravel.py`、`persistence.py`、`cli.py` | 已有 `/undo`，后续加 `checkpoint_label`/`fork_parent` 表或列；`_SCHEMA_VERSION=3` 分支迁移 | 安全。保持 persistence 懒加载；`graph.py` 顶层不 import | 老 v2 DB 自动迁移；pending shell 时继续拒绝 rewind | S/M |
| tenant 隔离真正生效 | `schema.py`、`persistence.py`、`sql_interact.py`、`registry.py`、`cli.py` | `thread_meta/audit_event` 增 `tenant_id`；list/query/delete 按 tenant 过滤；SQL DSN/cache key 包含 tenant；工具 tag 可按 tenant policy 过滤 | 中等风险。`ContextSchema.tenant_id` 已有；不能污染 `SearchState` | 两个 `ASKANSWER_TENANT_ID` 进程共享 DB，互不可见 threads/audit | L |
| 可观测性三路线分层 | `load.py`、`audit.py`、`_react_internals.py`、可选 `telemetry/` | LangSmith 作为外部 trace；OpenTelemetry 作为标准 exporter；自建 audit 继续做本地治理和 `/usage` | 安全。监控数据不要写 `SearchState`；不要缓存内部 model backend | 同一轮请求能在 audit 查 token/tool，在 LangSmith 或 OTEL 看到 trace id | M/L |
| `thread_meta` 并发窗口收敛 | `persistence.py` | 将 `upsert_meta` 改成 SQLite UPSERT；必要时给 `updated_at` 使用 max/current；删除与更新保持同事务 | 安全。WAL/busy_timeout 已有，但 SELECT 后 UPDATE 仍是逻辑竞争点 | 两进程同 thread_id 并发写 100 次，无丢 title/preview 异常 | S |

### 线 4：交互与产品形态

| 方向 | 触达模块 | 抽象改动 | 风险与不变量影响 | 验证方式 | 工作量 |
|---|---|---|---|---|---|
| HTTP/WebSocket server | 新 `askanswer/server.py`、抽 `cli.py` 的 command/service 层 | 把 `stream_query` 拆成 graph runner + renderer；HTTP 返回 events，WebSocket 传 token/tool/interrupt | 安全但中大型。不能把 CLI UI 逻辑塞进 graph；graph 仍复用 | 用同一 `create_search_assistant()`，CLI 与 WS 输出同一事件序列 | L |
| 通用澄清能力 | `intents/base.py`、`nodes.py`、`helix/nodes.py`、`ui_select.py` | 新 `ClarificationRequest` 协议；handler 可声明 `clarify(state)`；TTY 用 menu，非 TTY 用 default | 中等风险。不能让父图拓扑按 intent 分叉；建议作为 answer 前的工具/子图能力 | file_read 缺路径、sql 缺 DSN、research 范围不清三类都能澄清 | M |
| 轻量 Web UI 浏览历史 | 新 `askanswer/web.py` 或静态页 + JSON endpoint、`persistence.py` | 只读展示 `/threads`、messages、audit、checkpoints；先不做在线聊天 | 安全。复用 persistence；不要引入新外部存储 | 打开页面浏览历史、审计、导出；与 CLI 删除结果一致 | M |

## 3. 价值-可行性矩阵

|  | 高可行性 | 低可行性 |
|---|---|---|
| 高价值 | `thread_meta` 并发收敛；`/undo` schema v3 加固；MCP 健康与 profile；`fs_write` 确认类；`research_brief` | tenant 隔离真正生效；HTTP/WebSocket server；通用澄清能力；可观测性 LangSmith/OTEL 分层 |
| 低价值 | `decision_memo`；Helix 作为 MCP server；轻量 Web UI 浏览历史 | `code_review` 子图；`data_profile` 子图；`external_api_paid` 确认类 |

## 4. Top 3 推荐

### 1. `fs_write` 确认类落地

这是 TODO 里“统一 HITL 入口”的直接补充，不冲突。当前 registry 已有 `fs_write` 枚举但 executor 未接入，做它能把工具治理从 shell 扩展到真实副作用工具，价值高且边界清楚。

### 2. tenant 隔离真正生效

这是对 `ContextSchema.tenant_id` 的补全，也是企业化挑战文档中多租户路线的核心。它不是替代 TODO，而是持久化、SQL、工具治理三条线的企业化升级。

### 3. MCP 健康与 profile 连接

这是 TODO P2 “MCP 预热 & 健康检查”的自然延伸。当前 `/mcp add_url/add_stdio` 与 registry refresh 已通，下一步应让外部工具生态更可运营，而不是继续堆内置工具。

## 5. 放弃考虑的方向

| 放弃方向 | 理由 |
|---|---|
| 引入 Redis/队列系统 | 仓库没有相关依赖；用户约束要求不假设外部依赖。 |
| 改主图为按 intent 分支拓扑 | 明确违反 `understand -> answer -> sorcery` intent-agnostic 不变量。 |
| 让工具绕过 `ToolRegistry` 直接 `bind_tools` | 破坏统一治理、确认类和 tag 过滤。 |
| 在 `SearchState` 里塞 telemetry/metrics 大对象 | 违反状态语义，会膨胀 checkpoint；监控应走 audit/bus。 |
| 把 `tenant_id` 写进每条 message | 运行时配置已在 `ContextSchema`，应在 persistence/query 层隔离。 |
| 直接给 SQL agent 加写操作 | 现有源码已加只读拦截，企业化方向应继续强化安全，不应扩写权限。 |
| 做完整企业 Web 管理台 | 当前更适合先做只读历史 UI；完整 RBAC/SSO/管理端会超出仓库现有形态。 |
| 重做 Helix 为父图节点 | 已按 subgraph-as-tool 落地，改父图节点会破坏既有设计目标。 |
