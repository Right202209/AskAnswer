# 企业级"持久化与会话管理"增强（CLI 单用户场景）

## Context

计划起草时，`askanswer` 的会话状态完全在 `InMemorySaver` 里，CLI 进程退出即清零；用户无法回到昨天的对话、无法回退某一步重跑、无法导出历史、无法回看到底是哪些工具被调用过、消耗了多少 token。

当前实现已经切到 SQLite checkpointer，并完成本计划的阶段 A/B/C。阶段 D（静态加密）仍按原设计保留为可选增强。

TODO.md 已经把"持久化 checkpointer / `/resume` / `/threads` / `/undo`"列为 P0/P1，但只点到 `SqliteSaver` 一句话，没设计配套的元数据表、CLI 命令、审计与配额。本计划在保持"CLI 单用户"前提下，把这一块做成"企业级"——可恢复、可追溯、可导出、可清理、可加密。

不包含（用户已排除）：HTTP API、多租户、SSO、RBAC、Web UI。

---

## 范围与分阶段

**阶段 A · 持久化 + 线程管理（必做，单 PR）**
**阶段 B · 时间旅行 / 检查点（必做，单 PR）**
**阶段 C · 审计日志 + Token/成本计量 + 导出（推荐，单 PR）**
**阶段 D · 静态加密（可选）**

每个阶段独立可发布、独立可回归。

## Implementation status

| 阶段 | 状态 | 已落地内容 |
| --- | --- | --- |
| A · 持久化 + 线程管理 | done | `SqliteSaver`、`thread_meta`、`/threads`、`/resume`、`/title`、`/delete`、`/status` 持久化信息 |
| B · 时间旅行 / 检查点 | done | `askanswer/timetravel.py`、`/checkpoints`、`/undo`、`/jump`、`/fork`，挂起 shell interrupt 时拒绝回退 |
| C · 审计日志 + Token/成本 + 导出 | done | `audit_event`、LLM callback、工具/shell/MCP/model 审计、`/audit`、`/usage`、`/export`、`/import` |
| D · 静态加密 | deferred | 仍为可选；阶段 A/B/C 稳定后再决定是否引入 `cryptography` 或 SQLCipher |

---

## 阶段 A · 持久化 + 线程管理

### 设计要点

1. **存储位置**：`~/.askanswer/state.db`（XDG 兼容：若设了 `XDG_DATA_HOME` 用 `$XDG_DATA_HOME/askanswer/state.db`）。目录创建在 `cli.main` 里完成。
2. **checkpointer**：`langgraph-checkpoint-sqlite` 的 `SqliteSaver.from_conn_string(...)`。注意它需要进入 context manager 一次后才返回 saver；用一个模块级的 `_PersistenceManager` 持有 conn 和 saver，`atexit` 关闭。
3. **线程元数据表**（SqliteSaver 自身只存 checkpoint 快照，缺少 title / tags / 预览）：在同一 SQLite DB 里另建一张
   ```sql
   CREATE TABLE thread_meta (
     thread_id TEXT PRIMARY KEY,
     title TEXT,                    -- 用户命名或首条问题截取
     tags TEXT,                     -- JSON array
     created_at INTEGER,
     updated_at INTEGER,
     message_count INTEGER,
     last_intent TEXT,
     model_label TEXT,              -- 创建时的 provider:spec
     preview TEXT                   -- 最后一条用户消息前 80 字符
   );
   CREATE INDEX idx_thread_meta_updated ON thread_meta(updated_at DESC);
   ```
4. **写入时机**：在 `cli.stream_query` 末尾收集 `final_answer` 时写一次 upsert（一次会话一行更新）。**不**改 graph 内部，避免污染 reducer 语义。
5. **首次自动 title**：第一次写入时取用户首条消息前 30 字符做 title，用户后续可 `/title` 改名。

### 新文件 `askanswer/persistence.py`

```python
# 单一管理者：SqliteSaver、thread_meta 表、所有线程级查询入口
class PersistenceManager:
    def __init__(self, db_path: Path): ...
    @property
    def checkpointer(self) -> SqliteSaver: ...

    # 线程 CRUD（操作 thread_meta + SqliteSaver）
    def upsert_meta(self, thread_id, *, title=None, intent=None,
                    model_label=None, preview=None, message_count=None) -> None: ...
    def list_threads(self, limit=50, query: str | None = None) -> list[ThreadMeta]: ...
    def get_meta(self, thread_id) -> ThreadMeta | None: ...
    def set_title(self, thread_id, title) -> None: ...
    def delete_thread(self, thread_id) -> bool:  # 同时清掉 SqliteSaver 里该 thread 的 checkpoints
        # SqliteSaver 没暴露 delete API → 直接 DELETE FROM checkpoints WHERE thread_id=?
        ...
    def close(self) -> None: ...

def get_persistence() -> PersistenceManager: ...     # 单例
def shutdown_persistence() -> None: ...               # cli.main 里 atexit 注册
```

### 修改 `askanswer/graph.py`

```diff
-from langgraph.checkpoint.memory import InMemorySaver
+from .persistence import get_persistence
...
-    memory = InMemorySaver()
-    app = workflow.compile(checkpointer=memory)
+    app = workflow.compile(checkpointer=get_persistence().checkpointer)
```

注意：`create_search_assistant()` 当前每次调用都新编译。`cli.main` 已经只调一次（`cli.py:main` 里），不会重复打开 SQLite，但要在 `persistence` 里用 `Lock` 防御并发。

### 修改 `askanswer/cli.py`

1. **欢迎框**：`/status` 已有，扩展显示 thread 总数 + 当前 thread 的 title。
2. **新斜线命令**（在 `handle_command` 路由里）：
   - `/threads [keyword]` — 列出最近 50 个，按 `updated_at` 倒序，列：序号、id 缩写、updated、msgs、title、preview。支持模糊匹配 title/preview。
   - `/resume <id_or_index>` — 把当前 `thread_id` 切换到目标 thread，重新进入 REPL 等输入；不重放历史，但下一条问题就用旧 thread 的 checkpointer 状态。`stream_query` 的 `config={"configurable": {"thread_id": thread_id}}` 已经是 langgraph 标准写法，无需改 graph。
   - `/title <name>` — 给当前 thread 命名。
   - `/delete <id_or_index>` — 删除 thread（含 checkpoint），删自己当前 thread 时同时 `/clear`。**必须二次确认**（沿用 shell HITL 那套 prompt）。
3. **`/clear` 行为**：原来只是开新 thread；现在仍然开新 thread，**不**删旧的（让用户用 `/delete` 显式删除）。
4. **`stream_query` 末尾**：收集到 `final_answer` 后调用 `get_persistence().upsert_meta(thread_id, ...)`。

### 关键复用

- `cli.stream_query` 里 `thread_id` 已经是参数，直接换值即可恢复（`cli.py:185`-`stream_query` 签名）。
- `_runtime_context` 与 `ContextSchema` 不动。
- `current_model_label()`（`load.py:82`）用于元数据里记录创建时模型。

### 依赖

`pyproject.toml` 加 `"langgraph-checkpoint-sqlite>=2.0"`。

---

## 阶段 B · 时间旅行 / 检查点

### 设计要点

利用 LangGraph 内建 `app.get_state_history(config)`（按时间倒序产出 `StateSnapshot`）+ `app.update_state(config, values, as_node=...)`（创建分叉）。

### CLI 新命令

- `/checkpoints` — 列出当前 thread 的所有 snapshot：`#0 (latest) answer · 12s ago · 4 msgs ...`，标出每一步是哪个 node、step 字段、message_count。
- `/undo [n]` — 取倒数第 n 个 snapshot（默认 1），用 `app.update_state(config, snapshot.values)` 覆盖回去；下一条问题从该快照继续。
- `/jump <index>` — 同 `/undo` 但显式。
- `/fork` — 在当前 snapshot 处开新 thread（复制最近 N 条 messages 到新 thread_id）。

### 实现位置

- 新增 `askanswer/timetravel.py`：`list_checkpoints(app, thread_id)`、`rewind_to(app, thread_id, index)`、`fork_thread(app, src_thread_id, persistence)`。
- 在 `cli.handle_command` 里路由这些命令。

### 关键风险

- `update_state` 创建的"新 checkpoint"会让 `get_state_history` 里多一条；UI 上要标"分叉"。
- `interrupt()` 中断态恢复后再 undo 行为复杂——MVP 里禁用：检测到 `pending_shell` 非空时拒绝 `/undo`。

---

## 阶段 C · 审计日志 + Token/成本 + 导出

### 审计表

同库新增：

```sql
CREATE TABLE audit_event (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  kind TEXT NOT NULL,                 -- llm_call | tool_call | shell_approve | shell_reject | mcp_connect | model_swap
  tool_name TEXT,
  args_summary TEXT,                  -- 截断 200 字符
  result_size INTEGER,
  model_label TEXT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  duration_ms INTEGER,
  intent TEXT,
  error TEXT
);
CREATE INDEX idx_audit_thread_ts ON audit_event(thread_id, ts DESC);
CREATE INDEX idx_audit_kind_ts ON audit_event(kind, ts DESC);
```

### 接入点

1. **LLM 调用**：在 `askanswer/load.py` 的 `_ModelProxy` 里包一个 callback handler（langchain 的 `BaseCallbackHandler.on_llm_end` 给出 `LLMResult.llm_output["token_usage"]`），把 `input_tokens`/`output_tokens` 推到一个 thread-local 队列；`cli.stream_query` 在每轮结束 flush 到 `audit_event`。这样不需改图。
2. **工具调用**：`askanswer/_react_internals.py:_tools_node` 已经是单点，前后包计时并通过 `audit.log_event(...)` 记录。`_run_with_confirmation`（shell HITL）单独记录 `shell_approve` / `shell_reject`。
3. **MCP 连接 / 模型切换**：`cli.handle_command` 里 `/mcp <url>` 与 `/model` 处直接写 audit。

### 成本估算

- 维护一张极小的"价格表" `askanswer/pricing.py`：`{"openai:gpt-4o": (input_per_1k, output_per_1k), ...}`，未知 model_label 时输出空成本但仍记 token。
- `/usage [--days 7] [--thread <id>]` — 聚合查询：按模型、按 thread、按工具的 token / 成本 / 调用次数。

### 导出

- `/export <id_or_index> [--format md|json] [--out path]`
  - `md`：渲染成 `# Title` + 每条 message 的 role + content（multi-tool messages 折叠）
  - `json`：dump 完整 messages（用 LangChain `messages_to_dict`）+ thread_meta + 该 thread 的 audit_event
- `/import <path>`：仅 JSON；恢复成新 thread_id（不覆盖现有 ID），并切换到导入后的会话。

### 保留策略（轻量，后续可选）

- 配置文件 `~/.askanswer/config.toml` 里 `[retention]` 段：`max_threads = 200`、`max_age_days = 0`（0 = 不过期）。
- `cli.main` 启动时跑一次 `persistence.purge_expired(...)`，软删除（先重命名 title 加 `[deleted]`，下次启动彻底清）以防误删。MVP 可先只支持 `max_threads`。
- 当前实现暂未自动清理历史，避免在没有配置确认的情况下删除用户会话；如需要清理，先使用 `/delete` 显式删除。

---

## 阶段 D · 静态加密（可选）

两条路线，按依赖容忍度二选一：

| 方案 | 依赖 | 优点 | 缺点 |
| --- | --- | --- | --- |
| SQLCipher（`pysqlcipher3` + 系统 sqlcipher） | 重 | 透明加密整库 | 跨平台分发困难 |
| 应用层加密 messages 字段 | `cryptography`（纯 Python） | 易部署 | 需自己处理迁移、index 失效 |

推荐 **方案 B 但仅加密 `audit_event.args_summary` 与 `messages` payload**：用 Fernet（AES-GCM），密钥默认存 `~/.askanswer/.key`（chmod 600），可被 `ASKANSWER_ENCRYPTION_KEY` 环境变量覆盖。SqliteSaver 的 `BaseCheckpointSaver.put` 内部把 `checkpoint` 序列化为 bytes 后写盘——可以包一层 `EncryptedSqliteSaver` 在序列化与 SQLite write 之间做 Fernet 加解密。MVP 阶段建议只在阶段 A/B/C 稳定后再上。

---

## 关键文件改动一览

| 文件 | 改动 | 阶段 |
| --- | --- | --- |
| `pyproject.toml` | 加 `langgraph-checkpoint-sqlite`、可选 `cryptography`、`tomli`(py<3.11) | A,D |
| `askanswer/persistence.py` | **新建**：`PersistenceManager`、`thread_meta` 表、`get_persistence` | A |
| `askanswer/graph.py` | `InMemorySaver` → `get_persistence().checkpointer` | A |
| `askanswer/cli.py` | 新斜线命令 `/threads /resume /title /delete /checkpoints /undo /jump /fork /export /import /usage /audit`；`stream_query` 末尾写 meta；`atexit` 注册 `shutdown_persistence` | A,B,C |
| `askanswer/timetravel.py` | **新建**：list/rewind/fork helpers | B |
| `askanswer/audit.py` | **新建**：写入 + 查询 audit_event | C |
| `askanswer/load.py` | `_ModelProxy` 注入 token-usage callback；`set_model` 触发 audit | C |
| `askanswer/_react_internals.py` | `_tools_node` / `_run_with_confirmation` 前后调 `audit.log_tool_call` | C |
| `askanswer/pricing.py` | **新建**：模型价格表 + 估算函数 | C |

---

## 验收

每个阶段都通过下面一组手动回归（无单测前提下）：

**阶段 A**
1. `askanswer "hello"` → 退出 → 再启动 → `/threads` 看到这条。
2. `/resume 1` → 输入"上一个问题问的是什么" → 模型应能结合历史回答。
3. `/title 测试线程` → `/threads` 看到名字。
4. `/delete 1` → 二次确认 → `/threads` 不再出现；`sqlite3 ~/.askanswer/state.db "select count(*) from checkpoints"` 同步减少。

**阶段 B**
1. 走一段 `chat` 对话 4-5 轮 → `/checkpoints` 看到 4-5 条 → `/undo 2` → 再问 → 行为基于第 2 条快照。
2. shell HITL 等待中 `/undo` 应被拒绝（带提示）。
3. `/fork` → `/threads` 多出一条；旧 thread 不变。

**阶段 C**
1. 提两个问题（一个走 `tavily_search`、一个走 `sql_query` 或 `check_weather`）→ `/audit` 看到对应记录 + token 数。
2. `/usage --days 1` 输出 token 与成本汇总，模型未知时成本列为空但 token 在。
3. `/export 1 --format md --out /tmp/x.md` → 文件可读 + 含完整对话 + 末尾审计摘要。
4. `/export 1 --format json --out /tmp/x.json` → `/import /tmp/x.json` → `/threads` 多出一条带 `[imported]` 前缀。

**阶段 D**
1. 启动一次后 `file ~/.askanswer/state.db` 应仍是 SQLite（方案 B）；用 `sqlite3` 直接查 `audit_event.args_summary` 应为 base64 密文，应用内 `/audit` 仍能读出明文。

---

## 落地建议

- **阶段 A** 单 PR，~250 行（新文件 ~150，cli 改动 ~80，graph 改动 ~5）。先合，因为它解锁后续所有阶段。
- **阶段 B** 单 PR，~150 行。
- **阶段 C** 单 PR，~400 行（最大），但内部解耦清晰。
- **阶段 D** 视需求决定，MVP 不做。

读完这份后如果同意整体方向，建议先只批准 **阶段 A**，落地稳定后再开 B/C 的 issue。
