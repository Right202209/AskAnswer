# 📊 AskAnswer 实时监控系统 — 实施计划

> 后台长期监控 + 实时显示系统/项目运行情况

## 一、目标与范围

| 维度 | 监控内容 |
|---|---|
| **系统层** | CPU / 内存 / 磁盘 / 网络（进程级 + 主机级） |
| **图执行层** | `understand → answer → sorcery` 各节点耗时、调用次数、retry 次数、当前活跃 `thread_id` |
| **工具层** | 每个工具（`tavily_search` / `read_file` / `sql_query` / `gen_shell_commands_run` / `mcp:*`）的调用计数、p50/p95 延迟、失败率 |
| **LLM 层** | 当前模型、累计 token 输入/输出、估算费用、最近 N 次请求耗时 |
| **MCP 层** | 各 server 在线状态、工具数、最近一次心跳延迟 |
| **SQL Agent** | 子图调用次数、命中 `MAX_SQL_QUERY_CALLS` 次数、慢查询 |
| **HITL** | 待确认 shell 命令队列、最近被拒绝/批准的命令 |

**非目标**：替代 LangSmith / Prometheus；先做单机轻量自带能力，后期可对接外部。

---

## 二、整体架构

```
┌──────────────────────────────────────────────────┐
│ askanswer 主进程                                  │
│  ┌──────────┐    ┌─────────────────────────┐     │
│  │ LangGraph│───▶│ Telemetry Hooks (新增)  │     │
│  │  nodes   │    │  - 节点装饰器            │     │
│  │  tools   │    │  - tool wrap callback   │     │
│  │  mcp     │    │  - LLM callback handler │     │
│  └──────────┘    └─────────────┬───────────┘     │
│                                ▼                  │
│                  ┌──────────────────────────┐     │
│                  │ MetricsBus (in-memory)   │     │
│                  │ - ring buffer (events)   │     │
│                  │ - counters / histograms  │     │
│                  └──────┬─────────────┬─────┘     │
│                         │             │           │
│              ┌──────────▼──┐    ┌─────▼──────┐    │
│              │ SQLite Sink │    │ UDS / Pipe │    │
│              │ (可选持久化) │    │  广播      │    │
│              └─────────────┘    └─────┬──────┘    │
└──────────────────────────────────────┼────────────┘
                                       │
        ┌──────────────────────────────┴───────────────┐
        ▼                                              ▼
┌───────────────────┐                       ┌──────────────────┐
│ 内嵌 TUI          │                       │ 独立监控进程      │
│ /monitor 切到子屏  │                       │ askanswer-watch  │
│ rich.live 渲染    │                       │ 另开终端订阅       │
└───────────────────┘                       └──────────────────┘
```

**关键决策**：
1. **采集与显示解耦** — `MetricsBus` 居中，hooks 只 publish，前端 pull/subscribe。
2. **后台长期** = 主进程一直在采，落地到 `~/.askanswer/metrics.db` (SQLite WAL)；独立监控进程 tail 该 DB 即"长期"。
3. **实时**优先用内存 ring buffer + asyncio queue，SQLite 仅做 5s flush + 长期归档。

---

## 三、模块设计（建议新文件）

### 1. `askanswer/telemetry/__init__.py`（新）
导出 `bus`, `record_node`, `record_tool`, `record_llm`, `MetricEvent` dataclass。

### 2. `askanswer/telemetry/bus.py`（新）
```python
class MetricsBus:
    def __init__(self, ring_size=2000): ...
    def emit(self, event: MetricEvent): ...
    def snapshot(self) -> Dashboard: ...   # 聚合给 TUI
    def subscribe(self) -> AsyncIterator:  # 给独立进程
```
- 单例（`get_bus()`），线程安全（`threading.Lock`）。
- 字段：`ts, kind ('node'|'tool'|'llm'|'mcp'|'sys'), name, duration_ms, status, attrs`。

### 3. `askanswer/telemetry/hooks.py`（新）
- `wrap_node(name)`：装饰器，包 `nodes.py` 里 `understand_query_node` / `sorcery_node`。
- `TelemetryToolCallback`：实现 `BaseCallbackHandler`，挂到 `_tools_node`（`_react_internals.py`）里执行前后。
- `LLMUsageCallback`：挂到 `model.bind_tools(...).ainvoke(callbacks=[...])`，捕获 `usage_metadata`。

### 4. `askanswer/telemetry/sysinfo.py`（新）
基于 `psutil` 的轻量轮询线程：500ms 采集一次进程 RSS / CPU%、主机负载，写入 bus。

### 5. `askanswer/telemetry/sink_sqlite.py`（新）
后台 worker：每 5s 把 ring buffer 中尚未持久化的事件批量写入 `~/.askanswer/metrics.db`。表结构：
```sql
events(ts, kind, name, duration_ms, status, attrs_json)
sessions(thread_id, started_at, last_active, intent_history)
```

### 6. `askanswer/telemetry/dashboard.py`（新）
`rich.live.Live` 渲染 `Layout`：
- 顶栏：模型 / cwd / 当前 thread / 系统 CPU&MEM
- 中部 4 个 Panel：节点表、工具表、LLM 用量曲线、MCP server 状态
- 底栏：最近 10 条事件流

### 7. `askanswer/cli_monitor.py`（新，独立 entrypoint）
`pyproject.toml` 新增 `askanswer-watch = "askanswer.cli_monitor:main"`，从 SQLite 读取近 5 分钟数据并实时刷新 TUI。可与主进程同时跑、跑在另一台终端。

---

## 四、改动既有文件清单

| 文件 | 改动 |
|---|---|
| `askanswer/state.py` | 新增 `started_at: float`, `node_timings: dict` 字段（可选，仅当需要给单次会话画甘特图时） |
| `askanswer/_react_internals.py` | `_answer_node` / `_tools_node` 入口与出口插 `bus.emit`；LLM 调用挂 `LLMUsageCallback` |
| `askanswer/nodes.py` | `understand_query_node` / `sorcery_node` 加 `@wrap_node` |
| `askanswer/sqlagent/sql_node.py` | `run_query` 节点埋点（重点关注 `_trim_observation` 触发的截断率） |
| `askanswer/mcp.py` | `MCPClientManager` 新增 `health_check()` 协程，30s 一次 list_tools 探测；结果走 bus |
| `askanswer/cli.py` | `handle_command` 增加 `/monitor` 进入内嵌 TUI（按 `q` 返回 REPL）；`status_block` 加一段简短指标摘要 |
| `askanswer/registry.py` | `_seed_builtin` 注册一个新工具 `query_metrics`（bundle=chat），让 LLM 自己能"自查"运行状态 |
| `pyproject.toml` | 依赖加 `psutil>=5.9`；`rich` 已在 |

---

## 五、分阶段交付（建议 4 个迭代）

### Phase 1 · MVP（~1 天）
- [ ] `telemetry/bus.py` + `telemetry/hooks.py`
- [ ] 在 `_react_internals._tools_node` 与 `_answer_node` 埋 4 个埋点
- [ ] `cli.py` 新增 `/monitor` 子命令，先用纯文本版（`status_block` 扩展），不上 `rich.live`
- **里程碑**：跑一次 `askanswer "查天气"`，`/monitor` 能看到 1 条 LLM、1 条 tool、3 条 node 记录

### Phase 2 · 实时 TUI 与系统指标（~1 天）
- [ ] 接入 `psutil` + `sysinfo.py` 后台线程
- [ ] `dashboard.py` 用 `rich.live.Live` + `Layout` 渲染
- [ ] LLM token 计数（OpenAI / Anthropic 都有 `usage_metadata`，加个最简价格表 dict）
- **里程碑**：`/monitor` 进入 1Hz 刷新的全屏面板，按 `q` 退出回 REPL

### Phase 3 · 持久化 + 独立监控进程（~1.5 天）
- [ ] `sink_sqlite.py` 落库（SQLite WAL 模式，避免与未来 `SqliteSaver` checkpointer 冲突 — 用独立 db 文件 `metrics.db`）
- [ ] `askanswer-watch` 命令（独立 entrypoint）
- [ ] MCP 健康探测线程
- **里程碑**：另开一个终端跑 `askanswer-watch`，能实时看到主进程的所有事件，主进程退出后历史仍可回看

### Phase 4 · 进阶（可选，~按需）
- [ ] `query_metrics` 工具（让 LLM 自己回答"过去 1 小时哪个工具最慢？"）
- [ ] LangSmith export adapter（与 P0-04 联动）
- [ ] HTTP `/metrics` Prometheus 端点（启动 flag `--metrics-port 9100`）
- [ ] 异常告警：连续失败/p95 突刺时 REPL 顶部插入红条提示

---

## 六、风险与注意事项

1. **`load.py` 的 `_ModelProxy` 热替换** — `LLMUsageCallback` 不要缓存 `model._backend`，每次调用都从 proxy 取，否则 `/model` 切换后丢监控。
2. **`mcp.py` 的专用 asyncio 循环** — 健康探测必须 `asyncio.run_coroutine_threadsafe` 提交到 manager 的 loop，不能用主线程 loop。
3. **`interrupt()` 流的不可重入** — Phase 1 在 `_tools_node` 埋点要区分"正常返回"和"interrupt 抛出"两条路径，避免漏埋后者。
4. **不要写入 `SearchState`** — 监控数据走 bus，不污染消息合并状态（违反 CLAUDE.md 约定）。
5. **SQLite 与未来的 SqliteSaver checkpointer 隔离** — 用不同 db 文件，避免锁竞争。
6. **`gen_shell_commands_run` 的隐私** — 落库时对 `command` 字段做长度截断或脱敏开关（env `ASKANSWER_METRICS_REDACT_SHELL=1`）。
7. **REPL TUI 互斥** — `rich.live.Live` 启动期间禁用 stdin 输入框；退出时恢复。
8. **依赖** — 仅引入 `psutil`，不上 textual / prometheus_client（保持轻量；后者放 Phase 4 可选）。

---

## 七、可观测的核心指标（DoD）

实施完成后，`/monitor` 主面板必须能在不离开 REPL 的情况下回答：

1. **当前模型是什么？过去 5 分钟用了多少 token、估算花了多少钱？**
2. **哪个工具调用最慢？失败率多少？**
3. **最近一次 SQL agent 是否撞到 `MAX_SQL_QUERY_CALLS=2` 上限？**
4. **MCP server 现在是不是都活着？最久没响应的是哪个？**
5. **进程 RSS 是否在涨？是不是有内存泄漏？**

---

## 八、相关文件参考

- 主图：`askanswer/graph.py` — `START → understand → answer → sorcery`
- 状态：`askanswer/state.py` — `SearchState` TypedDict
- React 子图内部：`askanswer/_react_internals.py` — 节点体（埋点重点）
- 工具注册：`askanswer/registry.py` — `ToolDescriptor` + bundles
- MCP 管理：`askanswer/mcp.py` — `MCPClientManager` 后台 asyncio loop
- CLI 入口：`askanswer/cli.py` — `handle_command` 路由所有斜线命令
- 模型代理：`askanswer/load.py` — `_ModelProxy` 与 `set_model` 热替换

> 建议实施顺序：**先做 Phase 1 MVP**（约 200 行新代码 + 4 个埋点），跑通后再决定是否走 Phase 2 的 `rich.live` 全屏 TUI。
