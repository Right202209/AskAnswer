# AskAnswer 分步执行计划（Land → Harden → Evolve）

> 配套：`00-direction-and-goals.md`（为什么）、`01-spec.md`（不变量与质量门槛）。
> 本文是「压缩上下文后也能照着推进」的手册：按顺序执行，每步独立可提交、有验收。开始任一步前先 `git status` + 读本文件对应小节。
> 所有 `python`/`pip`/`askanswer` 命令先 `source .venv/bin/activate`。

## 进度总览（做完一项勾一项）

- [ ] A1 冒烟验证矩阵
- [ ] A2 安全清单审查
- [ ] A3 提交在途工作 + 本计划
- [ ] B1 pytest/ruff 基建
- [ ] B2 测试套件（≥25 用例）
- [ ] B3 lint 基线 + 文档回写
- [ ] B4 CI（如有远端；否则记 N/A）
- [ ] C1 拆分 cli.py（runner/renderer/commands）
- [ ] C2 通用澄清能力
- [ ] C3 HTTP/SSE server
- [ ] C4 只读 Web UI

依赖：A 内部顺序执行；B 依赖 A3；C1 依赖 B 全部；C3 依赖 C1；C4 依赖 C3。

---

## Milestone A · Land（落地在途工作）

### A1 冒烟验证矩阵（无 API key 即可跑）

对工作区当前状态逐项验证，任何一项失败先修复再继续：

1. `python -c "from askanswer.graph import create_search_assistant"` 导入成功。
2. `askanswer --graph -` 输出 mermaid，节点含 `understand/answer/sorcery/confirm_plan`。
3. 空库迁移：`ASKANSWER_DB_PATH=$(mktemp -d)/t.db` 下初始化 persistence，`PRAGMA user_version`（或版本表）= 4，存在 `thread_meta.tenant_id`、`audit_event.tenant_id`、`checkpoint_label` 表。
4. tenant 过滤：同一库写入 alice/bob 两条 thread_meta，`list_threads(tenant_id="alice")` 只见 alice；`tenant_id=None` 全见。
5. `mcp_profile`：save→load round-trip；写入是原子的（临时文件 + `os.replace`）。
6. telemetry 关闭态：不设 `LANGSMITH_API_KEY`/`ASKANSWER_OTEL_EXPORTER` 时 import 不创建 client、不报错。
7. `python -c "import askanswer.helix_mcp"` 不触发 persistence/graph（检查无 state.db 创建副作用）。
8. intents：注册表包含 8 个 handler（file_read, sql, helix, decision, math, research, search, chat）。

**验收**：8 项全过；失败项的修复计入 A3 同一批提交。

### A2 安全清单审查

对 `git diff` + 新增文件过安全规则（按 CLI 场景解释）：

- 无硬编码密钥/token（重点：`mcp_profile.py` 的 headers、telemetry 的 key 读取只从 env）。
- `write_file` 敏感路径拦截与大小上限仍生效；确认协议不可被工具本体绕过（不变量 7）。
- SQL 全部参数化（重点：persistence 新增的 tenant 过滤 SQL、`checkpoint_label` 查询）。
- audit 落库的参数摘要已脱敏（`external_api_paid`、mcp 事件）。
- 错误信息不泄露 DSN/密钥。

**验收**：逐项记录结论；发现 CRITICAL → 停下先修（安全响应协议），再继续。

### A3 提交

**策略**：在途 7 项功能在 `cli.py`/`registry.py`/`persistence.py` 上的 hunk 相互交织，按功能拆提交会制造大量「未经验证的中间树」。因此按「验证过的整体」提交：

1. `feat: land possibility-space phases 1.2-3.4` — 全部修改的跟踪文件 + `askanswer/{telemetry,research,decision}/` + `helix_mcp.py` + `mcp_profile.py` + `intents/{research,decision}.py` + `docs/possibility-space-*.md` + `CHANGELOG.md`（CHANGELOG 已如实记录该批内容）。
2. `docs: add plan-docs replan (direction, spec, execution plan)` — `plan-docs/`。

**验收**：`git status` 干净；两条提交符合 `<type>: <description>`，无 Co-Authored-By。

---

## Milestone B · Harden（安全网）

### B1 测试与 lint 基建

- `pyproject.toml` 加 `[project.optional-dependencies] dev = ["pytest", "ruff"]`；`pip install -e ".[dev]"`。
- 建 `tests/__init__.py` 不需要；`tests/conftest.py` 提供 fixture：临时 `ASKANSWER_DB_PATH`、隔离 env（清掉 TENANT/LANGSMITH/OTEL 变量）。
- 提交：`chore: add pytest+ruff dev tooling`。

### B2 测试套件（把 A1 矩阵固化 + 加深）

每文件一个主题，函数级用例；mock LLM/网络，不依赖 API key：

| 文件 | 覆盖 |
|---|---|
| `tests/test_persistence.py` | 空库建 v4；模拟 v2 旧库→迁移到 v4 数据无损；tenant 过滤（list/get/find/delete/audit/usage）；`upsert_meta` COALESCE 语义（None 不覆盖旧值）；`checkpoint_label` 增查 |
| `tests/test_mcp_profile.py` | round-trip、原子写、remove、损坏 JSON 容错（warning 不 crash） |
| `tests/test_confirmations.py` | 按 class 分发到对应 handler；fs_write 危险路径（`.env`/`*.pem`/`~/.ssh`/`/etc`/>1MB）全拒；协议 4 步顺序 |
| `tests/test_telemetry.py` | 无 env → 零副作用（不建 client/span）；有 env → 启用标志正确 |
| `tests/test_registry.py` | intent tag 过滤；`confirmation_class` 元数据；种子工具含 `research_brief_loop`/`decision_memo_loop`/`helix_spec_loop`/`sql_interact` |
| `tests/test_intents.py` | 8 handler 注册；`classify_local` 回退规则；`normalize` 字段清洗（非 search 清空 search_query 等） |
| `tests/test_graph.py` | mermaid 拓扑 golden 快照；import `graph` 不触发 persistence |

**验收**：`pytest -q` 全绿、≥25 用例。提交：`test: cover persistence, confirmations, mcp profile, telemetry, registry`（可按文件拆多条提交）。

### B3 lint 基线 + 文档回写

- `[tool.ruff]` target py310；先跑 `ruff check askanswer --statistics` 看噪声，选可全绿的规则集（至少 E/F/I），autofix + 少量手工修复；不做行为性改动。
- 回写 `.claude/mem/commands.md`：删除「没有测试/linter」的说明，写上 `pytest -q`、`ruff check askanswer`。
- 提交：`chore: ruff baseline` + `docs: record test/lint commands`。

### B4 CI

`git remote -v` 有 GitHub 远端才做：`.github/workflows/ci.yml` = ruff + pytest（Python 3.10/3.12 矩阵）。无远端则在本文件勾选框旁标 `N/A` 并说明。

---

## Milestone C · Evolve（形态演进，B 完成后逐个开工）

### C1 拆分 `cli.py`（2466 行 → 包）

目标结构：`askanswer/cli/` 包 —— `__init__.py`（main 入口，≤100 行）、`runner.py`（graph 事件流，纯逻辑无 UI）、`render.py`（rich 渲染）、`commands/`（slash 命令按域拆：threads/timetravel/mcp/audit/misc）。约束：纯移动+拆函数，不改行为；B2 测试全绿即验收；每个新文件 ≤300 行。提交：`refactor: split cli into runner/render/commands package`。

### C2 通用澄清能力

`intents/base.py` 加 `ClarificationRequest` 协议（`clarify(state) -> dict | None`）；answer 前分发，TTY 用 `ui_select` 菜单、非 TTY 用 default；不改父图拓扑。三场景验收：file_read 缺路径、sql 缺 DSN、research 范围不清。提交：`feat: generic clarification protocol`。

### C3 HTTP/SSE server

新 `askanswer/server.py` 复用 C1 的 runner；事件三类 token/tool/interrupt；CLI 与 HTTP 消费同一事件序列（用同一 `create_search_assistant()`）。确认类经 HTTP 时挂起并可 resume。提交：`feat: http sse server`。

### C4 只读 Web UI

静态页 + JSON endpoint：threads/messages/audit/checkpoints 浏览，按 tenant 过滤；不做在线聊天。提交：`feat: read-only web ui`。

---

## 恢复会话的快速 prompt

```
继续 plan-docs/02-execution-plan.md 的下一未勾选项。
先读该小节 + plan-docs/01-spec.md 的不变量，
git status 确认现状后再动手；完成后勾选进度框并按小节给出的信息提交。
```
