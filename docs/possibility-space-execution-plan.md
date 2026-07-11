# AskAnswer 可能性空间 · 分步执行计划

> 配套文档：`docs/possibility-space-exploration.md`（方向卡片与价值-可行性矩阵）。
> 本文件是“清理上下文后也能照着推进”的执行手册：每个阶段独立可提交，列出触达文件、改动要点、验收、需保护的不变量。
>
> **不变量（贯穿所有阶段，禁止破坏）**
> 1. 主图拓扑固定：`understand -> answer(react) -> sorcery`，禁止按 intent 拆分父图。
> 2. 工具必须经 `ToolRegistry`，禁止直接 `bind_tools` 绕过 tag/确认类治理。
> 3. `graph.py` 顶层禁止 import persistence/audit；`get_persistence()` 仅在 CLI 入口与 saver 注入处显式触发。
> 4. `SearchState` 只放消息/intent/control 字段，不塞 telemetry/metrics 大对象。
> 5. SQL agent 保持只读拦截，不新增写权限路径。
> 6. 节点返回 partial dict，永不返回完整 state。

## 阶段总览与依赖

```
Phase 0 (S)  thread_meta 并发收敛       —— 独立，随时可做
Phase 1 (M)  fs_write 确认类            —— 独立
Phase 1 (M)  MCP 健康与 profile        —— 独立
Phase 1 (L)  tenant 隔离真正生效        —— 依赖 Phase 0 完成（共享 persistence 改动面）
Phase 2 (M)  /undo schema v3 加固      —— 依赖 Phase 0
Phase 2 (L)  可观测性三路线分层         —— 独立（audit 已具雏形）
Phase 3 (M)  research_brief 子图        —— 独立
Phase 3 (M)  Helix 作为 MCP server      —— 独立
Phase 3 (S)  decision_memo 子图         —— 独立（复用 helix）
Phase 3 (M)  code_review / data_profile —— 低价值，可延后
Phase 3 (M)  external_api_paid 确认类   —— 依赖 Phase 1.1（共享确认类执行层）
Phase 4 (L)  HTTP/WebSocket server      —— 依赖 Phase 1.1/1.3 稳定
Phase 4 (M)  通用澄清能力                —— 独立
Phase 4 (M)  轻量 Web UI                 —— 依赖 Phase 1.3（按 tenant 过滤）
```

每个阶段建议拆为独立 PR；按上面顺序提交，前置依赖未落地时不开后续阶段。

---

## Phase 0 · 热身：`thread_meta` 并发 UPSERT 收敛

**目标**：消除 `upsert_meta` 中 `SELECT 1 → INSERT/UPDATE` 的逻辑竞争窗口，多进程同 thread 高频写入时不丢 title/preview。

**触达文件**
- `askanswer/persistence.py`（仅 `upsert_meta` 方法）

**改动要点**
1. 将“先 SELECT 再分支”改为 SQLite UPSERT：
   ```sql
   INSERT INTO thread_meta(thread_id, title, tags, created_at, updated_at,
                           message_count, last_intent, model_label, preview)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
   ON CONFLICT(thread_id) DO UPDATE SET
     title         = COALESCE(excluded.title, thread_meta.title),
     last_intent   = COALESCE(excluded.last_intent, thread_meta.last_intent),
     model_label   = COALESCE(excluded.model_label, thread_meta.model_label),
     preview       = COALESCE(excluded.preview, thread_meta.preview),
     message_count = COALESCE(excluded.message_count, thread_meta.message_count),
     updated_at    = MAX(excluded.updated_at, thread_meta.updated_at);
   ```
2. `_derive_title(preview)` 只在 `title` 入参为空时作为 INSERT 的默认值。
3. `created_at` 仅 INSERT 阶段写入；UPDATE 分支不动它。

**验收**
- 起两个 Python 进程同 `thread_id` 并发 `upsert_meta` 100 次，最终 `title`/`preview` 不为 NULL，且 `message_count` 是最后一次显式赋值。
- 已有 v2 库直接复用，不需新 migration。

**注意**
- `with self._lock` 仍保留以保护 Python 端 thread safety（sqlite3 cursor 不是线程安全）。
- 不改 schema 版本号；改 schema 在 Phase 2.1。

---

## Phase 1 · Top 3 推荐

### 1.1 `fs_write` 确认类落地

**目标**：把统一 HITL 治理从 shell 扩展到“写文件”这一最常见副作用工具，按文件大小/路径展示 diff 并要求确认。

**触达文件**
- `askanswer/tools.py`（新增 `write_file_patch` / `propose_file_write`，注意 read_file 的边界已加固，参考其风格）
- `askanswer/registry.py`（注册 + `confirmation_class="fs_write"`）
- `askanswer/_react_internals.py`（`_route_from_answer` 与 `_tools_node` 加 `fs_write` 路由分支；新增 `_fs_write_plan_node` 或扩展 `_shell_plan_node` 通用化）
- `askanswer/cli.py`（UI 渲染：path / size / diff 摘要 / 敏感路径告警）
- 可选 `askanswer/state.py` 加 `pending_fs_write: dict` 字段（与 `pending_shell` 同模式）

**改动要点**
1. **工具签名**：`write_file_patch(path: str, content: str, mode: Literal["overwrite","append","patch"]="overwrite")`，返回字符串结果。
2. **plan 节点**：参考 `_shell_plan_node`，把待写入的 path/content/diff 存入 `state["pending_fs_write"]`，由 `interrupt({"type":"confirm_fs_write", ...})` 暴露给 CLI。
3. **危险检查**：拒绝列表（`.env`、`*.pem`、`~/.ssh/*`、`/etc/*`、超过 1 MiB 的写入），参考 `check_dangerous` 的做法在 `tools.py` 加 `check_fs_write_dangerous(path, size)`。
4. **CLI 渲染**：复用 `ui_input.py` 中现有确认 UI 模式；展示 path、操作类型、字节数、前后 diff 片段（用 `difflib.unified_diff` 截断）。
5. **审计**：新增事件 `kind="fs_write_approve"` / `"fs_write_reject"`，参数摘要包含 path + size，不写 content。
6. **路由**：`_route_from_answer` 中改为根据 tool 的 `confirmation_class` 分发到 `shell_plan` 或 `fs_write_plan`，最终都汇入 `tools` 节点。建议把 `_shell_plan_node` 改名为 `_plan_confirmation_node` 并按 class dispatch，避免节点爆炸。

**验收**
- LLM 提案写文件 → CLI 显示 path/diff，未确认前文件未变动。
- 拒绝写 `.env`、`~/.ssh/id_rsa`、`/etc/passwd`、>1 MiB 文件。
- `audit_event` 新增对应记录；`/audit` 命令能列出 `fs_write_*`。

**注意**
- 不让工具直接 `open(path,'w')` 绕过确认；执行体必须由 `_tools_node` 在 resume 后再调用 Python IO。
- `state` 字段命名与 `pending_shell` 保持一致风格。

### 1.2 MCP 健康与 profile 连接

**目标**：让 MCP server 列表可观测、可持久化、可在启动时自动重连。

**触达文件**
- `askanswer/mcp.py`（`_ServerEntry` 加 `last_checked: int`/`status: str`/`last_error: str | None`；新增 `health_check(name)` 方法）
- `askanswer/registry.py`（refresh_mcp 后保留 status 摘要，便于 CLI 展示）
- `askanswer/cli.py`(`/mcp` 命令补 `health` / `list -v`)
- 新增 `askanswer/mcp_profile.py`（profile 文件读写）或复用 `~/.askanswer/mcp.json`

**改动要点**
1. `_ServerEntry` 新增字段，由 `add_url`/`add_stdio` 初始化为 `status="connected"`、`last_checked=now`。
2. `health_check(name)`：在后台 loop 上调用 `entry.session.list_tools()`，超时 3s，更新 `status` 与 `last_error`。
3. CLI 命令 `/mcp health [name]`：调用 `manager.health_check()`，渲染表格（name / transport / status / tools / last_checked）。
4. **Profile 文件**：`~/.askanswer/mcp.json`，结构：
   ```json
   {"servers": [{"name":"...","transport":"stdio|streamable_http|sse","command":"...","args":[...],"url":"...","headers":{...}}]}
   ```
   - 启动时 `cli.py` 在 persistence 初始化后调用 `mcp_profile.load()`，逐项 `manager.add_*`，失败仅 warning，不阻塞启动。
   - `/mcp add_url` / `/mcp add_stdio` 成功后调用 `mcp_profile.save_entry(...)` 追加。
   - `/mcp remove` 时同步删除。
5. **registry refresh**：health 探测失败时把对应 `mcp:<server>` 工具从 registry 清掉（避免给 LLM 看到死工具），恢复时 refresh。

**验收**
- 启停一个本地 stdio MCP server，profile 内出现/消失；下次启动自动重连。
- `/mcp health` 显示状态；故意断开后 status 变 `disconnected`，工具列表里看不到 `<server>__*`。
- `/audit --kind mcp_connect` 能查到连接事件。

**注意**
- 不能让 mcp.py 顶层 import 任何 graph/persistence 代码。
- profile 持久化是文件级写入，写之前先 `mkdir`，原子写（先写临时文件再 `os.replace`）。
- 不在 SearchState 里塞 MCP 状态——读自 manager。

### 1.3 tenant 隔离真正生效

**目标**：让 `ContextSchema.tenant_id` 在 persistence / SQL / 工具治理三处都真正生效，两个 tenant 在同一 SQLite 文件下互不可见。

**前置**：Phase 0 完成（避免和 UPSERT 改动冲突）。

**触达文件**
- `askanswer/persistence.py`（schema 迁移 v3：thread_meta + audit_event 加 `tenant_id` 列，所有读写按 tenant 过滤）
- `askanswer/schema.py`（已有 `tenant_id`，不需改）
- `askanswer/cli.py`（`/threads`、`/audit`、`/usage`、`/load` 等命令按当前 tenant 过滤；命令行展示 tenant 角标）
- `askanswer/sqlagent/sql_node.py` 或 `sql_interact.py`（按 `(tenant_id, db_dsn)` 作为缓存 key；不同 tenant 不复用 connection cache）
- `askanswer/registry.py`（可选：扩展 tag 过滤，按 tenant policy 屏蔽某些工具，需新增 `_tenant_policy` 表/文件，第一版可只做 persistence 隔离）
- `askanswer/graph.py`(`stream_query` 把 tenant_id 写进 audit 上下文)
- `askanswer/audit.py`（事件落库时带 tenant_id）

**改动要点**
1. **Schema v3 迁移**：
   ```sql
   ALTER TABLE thread_meta ADD COLUMN tenant_id TEXT;
   ALTER TABLE audit_event ADD COLUMN tenant_id TEXT;
   CREATE INDEX idx_thread_meta_tenant_updated ON thread_meta(tenant_id, updated_at DESC);
   CREATE INDEX idx_audit_tenant_ts ON audit_event(tenant_id, ts DESC);
   ```
   把 `_SCHEMA_VERSION` 改为 3 并在 `_migrate` 加 `if current < 3` 分支。
2. **PersistenceManager API**：所有读方法新增 `tenant_id` 参数（默认 `None` 等价旧行为，避免破坏既有 CLI 调用）：
   - `list_threads(tenant_id=None, ...)`、`get_meta(thread_id, tenant_id=None)`、`find_by_prefix(prefix, tenant_id=None, ...)`、`delete_thread(thread_id, tenant_id=None)`、`list_audit_events(..., tenant_id=None)`、`usage_summary(..., tenant_id=None)`。
   - 内部 SQL 加 `AND (? IS NULL OR tenant_id = ?)`（两个占位都填 tenant_id），让 `None` 表示“不限 tenant”。
   - `upsert_meta` 与 `log_audit_event*` 新增必填 `tenant_id`，未传则取 `os.environ.get("ASKANSWER_TENANT_ID")`，再兜底 `None`。
3. **CLI 层**：在每条命令入口取 `tenant_id = os.getenv("ASKANSWER_TENANT_ID") or None`（cli.py:410 已有），把它传给 persistence 调用。`/threads` 渲染表头展示 `tenant`。
4. **SQL agent cache key**：把 `(tenant_id, db_dsn)` 作为缓存 key 而非 dsn-only，避免跨 tenant 共享同一个 engine/conn pool。
5. **跨 tenant 隔离测试用例**：
   ```bash
   ASKANSWER_TENANT_ID=alice askanswer "记笔记 1"
   ASKANSWER_TENANT_ID=bob   askanswer /threads   # 看不到 alice 的
   ASKANSWER_TENANT_ID=bob   askanswer /audit     # 看不到 alice 的
   ```
6. **回归**：未设 `ASKANSWER_TENANT_ID` 时，所有命令行为与现有 master 分支一致（tenant_id 为 NULL 视为同一桶）。

**验收**
- 上述跨 tenant 测试通过。
- SqliteSaver 的 `checkpoints` 表暂不加 tenant_id（langgraph 自管表，不动）；改为在 CLI 列出 thread 时按 `thread_meta.tenant_id` 过滤，从而间接屏蔽其他 tenant 的 checkpoint。
- 既有 v2 库迁移后 tenant_id 列为 NULL，行为不退化。

**注意**
- 不要把 tenant_id 塞进 message 内容；它是运行时上下文。
- 不要把 tenant_id 写进 `SearchState`；保持只读 ContextSchema 形态。
- SqliteSaver 内部表跨 tenant 用 `thread_id` 命名空间（`{tenant}:{uuid}`）的方案在第一版**不引入**，复杂度过高。第一版只在 thread_meta/audit_event 层做过滤。

---

## Phase 2 · 加固与可观测性

### 2.1 `/undo` 加固与 schema v3 收尾

**前置**：Phase 1.3 已合 schema v3。本阶段在同一版本内追加列或独立 v4。

**触达文件**
- `askanswer/persistence.py`（schema 迁移加 `thread_checkpoint_label` 表 或 `thread_meta` 的 `last_undo_at` 列）
- `askanswer/timetravel.py`
- `askanswer/cli.py`(`/undo` 命令文案与边界提示)

**改动要点**
1. 新增表 `checkpoint_label(thread_id, checkpoint_id, label, created_at)`，让 `/undo --label foo` 命名节点。
2. `pending_shell` 不为空时继续拒绝 rewind（现有逻辑保留）。
3. `/undo` 输出 affected message count / 回到的 checkpoint 摘要，便于回滚。

**验收**
- 老 v2/v3 库自动迁移；带 pending_shell 的 thread 仍拒绝 undo。
- `/undo --label` 与无 label 走同路径。

### 2.2 可观测性三路线分层

**目标**：把现有 audit（本地治理）与外部 trace 解耦，新增 LangSmith / OpenTelemetry exporter，不污染 `SearchState`。

**触达文件**
- 新建 `askanswer/telemetry/__init__.py`、`telemetry/langsmith.py`、`telemetry/otel.py`
- `askanswer/load.py`（如需在 model 实例化时挂 LangSmith client）
- `askanswer/_react_internals.py` 与 `askanswer/nodes.py`（每个 LLM/tool 调用前后发 telemetry 事件；不要新加 state 字段）
- `askanswer/audit.py`（保留为本地审计；扩展事件流通过 callback 而非 state）

**改动要点**
1. 环境开关：`ASKANSWER_OTEL_EXPORTER=...` / `LANGSMITH_API_KEY` 控制启用。
2. trace context：用 contextvars 在节点入口 push，节点出口 pop；不写入 state。
3. audit 与 telemetry 共享 event schema 的最小子集（kind / tool_name / duration_ms / tokens），但写入路径独立。
4. 文档：`docs/monitoring-plan.md` 已存在，把实施版本作为附录写入。

**验收**
- 同一轮请求：`/audit` 能查 tokens/工具；LangSmith 或 OTEL backend 能看到对应 trace。
- 关闭环境变量后 zero overhead（不创建 client）。

**注意**
- 不要在 `_react_internals` 顶层 import telemetry——按需 lazy import。
- 不缓存内部 `_ModelProxy` 引用到 telemetry 模块，避免 model swap 失效。

---

## Phase 3 · 生态扩展

### 3.1 `research_brief` 子图

**目标**：多源搜索 + 交叉核验 + 列引用的研究简报子图。

**触达文件**
- 新增 `askanswer/intents/research.py` 实现 `IntentHandler`
- 新增 `askanswer/research/__init__.py`、`research/agent.py`、`research/nodes.py`、`research/research_tool.py`
- `askanswer/intents/__init__.py` 注册 `ResearchHandler`
- `askanswer/registry.py` 用 `_seed_research`（参照 `_seed_helix`）注册 `research_brief_loop` 工具到 `chat|research|research_tool` tag

**改动要点**
1. **handler**：`bundle_tags = frozenset({"research","search","research_tool","tavily"})`，`max_retries=2`，`prompt_hint` 说明“先搜索多源 → 交叉核验 → 列引用”。
2. **subgraph**：`plan_queries -> search -> synthesize -> source_check`，最后输出含引用列表的 brief。
3. **tool 入口**：`research_brief_loop(topic: str, max_queries: int = 5) -> str`，内部 `tavily_search` 多轮调用。
4. **不绑死 Tavily**：通过 registry 调用 `tavily_search`，便于将来替换。

**验收**
- 问“最近某政策/产品变化”，audit 显示多次 `tavily_search`，回答末尾带 `[1] url ...` 引用块。

### 3.2 `decision_memo` 子图

**目标**：澄清目标/约束/方案 → 输出 tradeoff memo；复用 Helix interview 节点作前置澄清。

**触达文件**
- 新增 `askanswer/intents/decision.py`
- 新增 `askanswer/decision/agent.py` 或直接在 helix 子图基础上派生
- `intents/__init__.py` 注册

**改动要点**
1. handler `max_retries=0`（Helix 自评估已覆盖，避免父 sorcery 再重试）。
2. 非 TTY 下用 `default_answer` 兜底。
3. 不修改父图。

**验收**
- TTY 下可交互澄清；非 TTY 跑 default_answer 流程不卡。

### 3.3 `code_review` / `data_profile` 子图（低价值，可延后）

按方向卡片操作；要点：
- `code_review`：通过 `read_file` + 摘要走 `collect_files -> analyze -> rank_findings -> finalize`，禁止直接递归读大目录（需在工具入口做大小/数量限流）。
- `data_profile`：复用 SQL agent 的只读约束，不开 DML 路径；CSV 走 `read_file` 采样。

### 3.4 Helix 作为 MCP server 暴露

**触达文件**
- 新增 `askanswer/helix_mcp.py`（MCP server 入口）
- 不改 `helix/agent.py`，只 reuse `run_helix_agent`

**改动要点**
1. 暴露 tool `helix_spec_loop(topic)`，传给外部 MCP client。
2. **禁止顶层 import** `graph.py`，避免触发 persistence 初始化。
3. 非 TTY 下回退 default_answer。

### 3.5 `external_api_paid` 确认类

**前置**：Phase 1.1 把确认类执行层通用化（`_plan_confirmation_node` dispatch）。

**触达文件**
- `askanswer/registry.py`（`ToolDescriptor` 加 `cost_estimate`/`risk_class` 元数据可选）
- `askanswer/_react_internals.py`（dispatch `external_api_paid` 到新的 plan 节点：展示 host/cost/params 摘要）
- `askanswer/cli.py`（渲染）
- `askanswer/mcp.py`（注册 MCP tool 时若 server 声明付费/写副作用，自动打 `confirmation_class="external_api_paid"`，需要约定 server-side hint）

**改动要点**
- 第一版只做 CLI 提示 + 审计脱敏；不做实际计费集成。
- args_summary 写入 audit 时必须做参数脱敏（去 token/email 等）。

---

## Phase 4 · 形态演进（可选）

### 4.1 HTTP/WebSocket server

**前置**：Phase 1.1（fs_write 确认通用化）+ Phase 1.3（tenant 稳定）。

**触达文件**
- 新增 `askanswer/server.py`
- 抽 `cli.py` 的 stream renderer 出来到 `askanswer/render.py` 或 `cli/render.py`
- `askanswer/graph.py` 保持不变（被 server 和 cli 共用）

**改动要点**
- 拆 `stream_query` 为 graph runner（产生事件流）+ renderer（CLI vs HTTP/WS）。
- WebSocket 传 token/tool/interrupt 三类事件；HTTP 走 SSE 或返回完整事件序列。
- 不把 CLI UI 逻辑塞进 graph 节点。

### 4.2 通用澄清能力

**触达文件**
- `askanswer/intents/base.py`（新增 `ClarificationRequest` Protocol：`clarify(state) -> dict|None`）
- 各 `intents/*.py` 选择性实现
- `askanswer/nodes.py`（answer 前增加 clarification dispatch；不改父图拓扑）
- `askanswer/ui_select.py`（TTY menu）

**改动要点**
- 严禁让父图按 intent 分叉拓扑；澄清作为 answer 前的可选工具/子能力。
- 三类场景验收：`file_read` 缺路径、`sql` 缺 DSN、`research` 范围不清。

### 4.3 轻量 Web UI 浏览历史

**前置**：Phase 1.3 完成。

**触达文件**
- 新增 `askanswer/web.py`（FastAPI 或 starlette 起一个只读端点）
- 静态页面 `askanswer/web/static/`
- `askanswer/persistence.py`(`list_threads`/`list_audit_events` 已经够用)

**改动要点**
- 只读：threads、messages、audit、checkpoints；不做在线聊天。
- 按 tenant 过滤；通过 `ASKANSWER_TENANT_ID` 或登录态。

---

## 放弃方向（不开 PR）

参考 `docs/possibility-space-exploration.md` §5：
- Redis/队列、按 intent 分支拓扑、绕过 ToolRegistry、状态膨胀、message 携带 tenant、SQL 写权限、完整企业管理台、Helix 回退为父图节点。

## 提交模板

每个阶段建议的 commit 前缀：

| 阶段 | 前缀示例 |
|---|---|
| 0.1 | `fix(persistence):` |
| 1.1 | `feat(security): fs_write confirmation class` |
| 1.2 | `feat(mcp): health checks + profile autoconnect` |
| 1.3 | `feat(persistence): tenant_id propagation` |
| 2.1 | `feat(persistence): undo labels + schema v3 follow-up` |
| 2.2 | `feat(telemetry): langsmith/otel exporters` |
| 3.x | `feat(intent): <name>` |
| 4.x | `feat(server|web|clarify): ...` |

## 启动新会话的快速 prompt

```
继续 docs/possibility-space-execution-plan.md 的 Phase <X.Y>。
先读该文件对应章节的“触达文件”和“改动要点”，
再执行 git status / 读相关文件确认当前状态后再动手。
保持文件顶部列出的不变量。
```
