# 架构重构落地状态（对照《架构重构设计文档》）

> 约束：**项目名与包名保持 `askanswer` 不变**。设计文档中的 `mazal/...` 路径一律映射到
> `askanswer/...`，不做任何改名。设计文档的 `skills/` 对应本项目已有的 `intents/`。

## 1. P1–P14 逐条状态

| # | 问题 | 处理 | 状态 |
|---|---|---|---|
| P1 | 导入时副作用 | 导入不再改写环境变量、不建客户端；入口显式 `askanswer.config.ensure_environ()`；模型标签 / 工具限额运行时读取 | ✅ |
| P2 | 全局可变单例 | `runtime/container.py` 组装根（tools / intents / models / confirmations / settings / store / events，均惰性）；全部节点由 `make_xxx_node(container)` 生成；intent 的 `evaluate()` 接收注入的模型；测试经 Container 注入替身 | ✅（进程默认实现仍作为 `build_container()` 的缺省值存在） |
| P3 | 配置靠改写 `os.environ` | Pydantic `Settings`；只有 `config/loader.py` 读 `ASKANSWER_*`（测试守护）；密钥 `SecretStr` | ✅（`$EDITOR`、LangSmith 等第三方约定变量除外） |
| P4 | 状态臃肿 | `SearchState` = 通用字段 + `slots`；`step` 为 `Phase` 枚举；删除死字段 | ✅ |
| P5 | 扩展点不彻底 | 分类提示由注册表按 handler `description` 生成 | ✅ |
| P6 | 进度提示混进历史 | 全部移除；进度由 CLI 按节点 update 渲染 | ✅ |
| P7 | 评估靠解析文本 | `with_structured_output(SearchVerdict)` | ✅ |
| P8 | 工具内藏 LLM 调用 | 工具 / 规划不再调 LLM | ✅ |
| P9 | HITL 恢复重复执行 | `plan_approvals → ask_approval ⟲ → act` | ✅（回归测试） |
| P10 | 外部请求无超时 | 统一显式超时（`Settings.limits`） | ✅ |
| P11 | 持久化职责过多 | `storage/db.py` + `storage/repos/{threads,audit,labels}` + `Store` 门面；删除走 `SqliteSaver.delete_thread` | ✅ |
| P12 | 流式未按节点过滤 | 只放行 `reason` 节点 token | ✅ |
| P13 | 工具标签与技能名混用 | `risk` / `capability`；高风险工具强制声明审批；`ASKANSWER_DENY_TOOLS` | ✅ |
| P14 | 目录平铺 | 分层目录（见 §3），包名不变 | ✅ |
| D1 | 父图 + react 子图 | 单层扁平 7 节点 + golden Mermaid 测试 | ✅ |
| — | 事件总线 | `observability/bus.py`；审计、telemetry 为订阅者 | ✅ |

## 2. §7.4 不可绕过的不变量

| 不变量 | 状态 |
|---|---|
| WRITE/EXEC/PAID 工具不声明审批无法注册 | ✅ `ToolDescriptor.__post_init__`，有测试 |
| act 执行审批项前二次检查 | ✅ 各 handler 的 `apply()` 内重新校验 |
| 会 interrupt 的节点无副作用 | ✅ `clarify` / `ask_approval`，有回归测试 |
| 子 Agent 拿不到需审批的工具 | ✅ 子图不绑定 shell / write_file |
| 非 TTY 审批默认拒绝 | ✅ `select_option` 非 TTY 返回默认项“取消” |
| 只有 config/loader 读 ASKANSWER_* 环境变量 | ✅ `tests/test_config.py` |
| 主图拓扑不被无意改动 | ✅ `tests/golden/graph.mmd` |
| 审计落库前脱敏 | 🟡 `redact_audit_args` 只用于付费 API 参数，其余仅记录 path/command |

## 3. 模块迁移对照（旧 → 新，均在 `askanswer/` 下）

| 旧 | 新 |
|---|---|
| `graph.py` / `react.py` / `_react_internals.py` / `nodes.py` / `answering.py` / `clarify.py` | `graph/builder.py`、`graph/edges.py`、`graph/nodes/{route,clarify,reason,approvals,act,review}.py` |
| `state.py` / `schema.py` | `graph/state.py` / `graph/context.py` |
| `load.py` / `routing.py` / `context.py` / `pricing.py` | `llm/models.py` / `llm/routing.py` / `llm/context.py` / `llm/pricing.py` |
| `settings.py` | `config/layers.py`（+ `config/{schema,loader,paths}.py`） |
| `registry.py` / `tools.py` | `tools/registry.py` / `tools/builtin.py` |
| `mcp.py` / `mcp_profile.py` | `mcp/manager.py` / `mcp/profile.py` |
| `confirmations.py` | `safety/approvals.py` |
| `persistence.py` | `storage/{db,models,store}.py` + `storage/repos/*` |
| `audit.py` / `telemetry/` | `observability/audit.py` / `observability/telemetry/`（+ `bus.py`） |
| `runner.py` / `timetravel.py` | `runtime/runner.py` / `runtime/timetravel.py` |
| `cli/` / `ui_input.py` / `ui_select.py` / `ui_spinner.py` | `interfaces/cli/`（`input.py` / `select.py` / `spinner.py`） |
| `server.py` / `wire.py` / `helix_mcp.py` | `interfaces/http/{server,wire}.py` / `interfaces/mcp_server/helix_server.py` |
| `sqlagent/` / `helix/` / `research/` / `decision/` | `agents/{sql,helix,research,decision}/` |
| 节点名 `understand` / `answer` / `tools` / `confirm_plan` / `sorcery` | `route` / `reason` / `act` / `plan_approvals` / `review` |

入口：`askanswer = askanswer.interfaces.cli:main`，`askanswer-server = askanswer.interfaces.http.server:main`，`python -m askanswer` 不变。

## 4. 仍可继续的工作（非阻塞）

- 审计全量脱敏（目前只覆盖付费 API 参数）。
- `intents/` 是否按设计文档改称 `skills/`：属于命名调整，本次按约束未改。
- 设计文档的 `ModelHub` / `RoutedModel` 拆分：现有 `llm/models.py` + `llm/routing.py` 已覆盖热切换与回退，未再细拆。

## 5. 验证

```
pip install -e '.[dev]'
pytest -q
ruff check askanswer tests
python -m askanswer --graph   # 无任何环境变量也能导出
```
