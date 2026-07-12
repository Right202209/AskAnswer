# AskAnswer 分步执行计划（Land → Harden → Evolve）

> 配套：`00-direction-and-goals.md`（为什么）、`01-spec.md`（不变量与质量门槛）。
> 本文是「压缩上下文后也能照着推进」的手册：按顺序执行，每步独立可提交、有验收。开始任一步前先 `git status` + 读本文件对应小节。
> 所有 `python`/`pip`/`askanswer` 命令先 `source .venv/bin/activate`。

## 进度总览（做完一项勾一项）

- [x] A1 冒烟验证矩阵（8 项全过 + v2→v4 旧库无损升级验证）
- [x] A2 安全清单审查（发现并修复：paid-API 审计参数未脱敏 → `redact_audit_args`）
- [x] A3 提交在途工作 + 本计划（89b7ea8 / e855cb8 / 6c94d42）
- [x] B0 修复无 key 导入阻塞 + 补齐缺失依赖（19355ff / 153427a）—— B 的前置，见下方说明
- [x] B1 pytest/ruff 基建（92ee5fa）
- [x] B2 测试套件（107 用例 / 7 文件，无 API key 可跑）（d8cebca）
- [x] B3 lint 基线 + 文档回写（9483ee6 / b1258b3）
- [x] B4 CI（GitHub Actions，ruff + pytest，py3.10/3.12）（f1409b2）
- [x] C1 拆分 cli.py（runner/renderer/commands）（本次：`askanswer/cli/` 包，16 模块均 ≤300 行；`stream_query` 收敛到 `runner.stream_leg`）
- [x] C2 通用澄清能力（a106a57）—— ⚠ 早于 C1 落地（偏离原定顺序）
- [x] C3 HTTP/SSE server（b0c862d）—— ⚠ 早于 C1 落地：runner.py 已抽出，但 cli.py 尚未成包
- [ ] C4 只读 Web UI（web/ 已有静态页，缺 JSON endpoint）

依赖：A 内部顺序执行；B 依赖 A3；C1 依赖 B 全部；C3 依赖 C1；C4 依赖 C3。
**执行偏离（如实记录）**：C2、C3 在 B2–B4 与 C1 之前就已提交（`runner.py` 为 C3 提前抽出，
但 `cli.py` 仍是 2510 行的单文件、未拆成包）。因此 C1 现在是「补做欠账」：既是 300 行上限的
存量债，也让 C3 的 runner 与 CLI 真正共用一套事件流。Milestone B 已回填完成并全绿。

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

> **状态：已完成并全绿**（`ruff check askanswer tests` + `pytest -q` = 107 passed）。
> 下面各小节保留原始设计意图，并在末尾补「实际落地」。

### B0 前置修复（B 开工时发现）

原计划假设「包可无 API key 导入」（spec §5 冒烟第 1 项），但 `load.py` 在**导入时**就
`init_chat_model` 构造 backend，缺 `OPENAI_API_KEY` 直接抛错——挡住了 B2/B4 与 CI。另外
`langchain-openai`、`langchain-community` 被 import 却未在 `pyproject` 声明（后者缺失时
`sql_query` 被静默跳过）。修复：

- `_ModelProxy` 惰性构造 backend（首次 invoke/stream/bind_tools 时才建；`/model` 热替换不变）。
- `pyproject` 补 `langchain-openai` / `langchain-community` 两条依赖。
- 提交：`19355ff`（惰性模型 + 去掉未用的 `OpenAI` import）、`153427a`（补 community 依赖）。

### B1 测试与 lint 基建

- `pyproject.toml` 加 `[project.optional-dependencies] dev = ["pytest", "ruff"]`；`pip install -e ".[dev]"`。
- 建 `tests/__init__.py` 不需要；`tests/conftest.py` 提供 fixture：临时 `ASKANSWER_DB_PATH`、隔离 env（清掉 TENANT/LANGSMITH/OTEL 变量）。
- 提交：`chore: add pytest+ruff dev tooling`。
- **实际落地**：`92ee5fa`（conftest 提供 `_isolated_env` autouse + `pm` 临时库 fixture）。

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
- **实际落地**：`d8cebca`，7 文件共 **107 用例**全绿；无 API key 运行（`sql_query` 依赖 community，已在 B0 补齐）。

### B3 lint 基线 + 文档回写

- `[tool.ruff]` target py310；先跑 `ruff check askanswer --statistics` 看噪声，选可全绿的规则集（至少 E/F/I），autofix + 少量手工修复；不做行为性改动。
- 回写 `.claude/mem/commands.md`：删除「没有测试/linter」的说明，写上 `pytest -q`、`ruff check askanswer`。
- 提交：`chore: ruff baseline` + `docs: record test/lint commands`。
- **实际落地**：`9483ee6`（E/F/I，忽略 E501；autofix 跨 36 文件仅动 import；手工把 `registry.py` 的 mid-file `import os` 上移）+ `b1258b3`（回写 `plan-docs/mem/commands.md`；`.claude/mem/commands.md` 同步更新但被 gitignore）。

### B4 CI

`git remote -v` 有 GitHub 远端才做：`.github/workflows/ci.yml` = ruff + pytest（Python 3.10/3.12 矩阵）。无远端则在本文件勾选框旁标 `N/A` 并说明。
- **实际落地**：远端存在（`git@github.com:Right202209/AskAnswer.git`）→ `f1409b2` 加 `.github/workflows/ci.yml`（push/PR to master，py3.10/3.12 矩阵，`pip install -e ".[dev]"` → ruff → pytest）。

---

## Milestone C · Evolve（形态演进，B 完成后逐个开工）

### C1 拆分 `cli.py`（2510 行 → 包）

目标结构：`askanswer/cli/` 包 —— `__init__.py`（main 入口，≤100 行）、`runner.py`（graph 事件流，纯逻辑无 UI）、`render.py`（rich 渲染）、`commands/`（slash 命令按域拆：threads/timetravel/mcp/audit/misc）。约束：纯移动+拆函数，不改行为；B2 测试全绿即验收；每个新文件 ≤300 行。提交：`refactor: split cli into runner/render/commands package`。
- **注意（当前状态）**：C3 已提前抽出顶层 `askanswer/runner.py`（graph 事件流），CLI 与 HTTP server 都消费它；但 `cli.py` 仍是 2510 行单文件。C1 = 把 `cli.py` 拆成 `askanswer/cli/` 包并复用已有 `runner.py`（避免再造一份事件流），render/commands 从 `cli.py` 迁出。B2 的 107 个测试是这次纯移动重构的回归网。
- **实际落地**：`askanswer/cli/` 包（16 模块，全部 ≤300 行；`__init__.py` 92 行含 `main` + 向后兼容再导出）。模块：`theme`/`text`/`render`/`progress`/`confirm`/`stream`/`repl`/`app` + `commands/`（`__init__` 路由 + `model`/`mcp`/`mcp_view`/`threads`/`timetravel`/`audit`/`transfer`/`_common`）。`stream_query` 收敛到 `runner.stream_leg`（删掉 CLI 自持的 `app.stream` 事件循环与 `_handle_message_chunk`/`_extract_interrupt_value`/`_pending_interrupt`——逻辑已在 runner 里），turn 级记账仍留在 CLI（不改审计粒度）。新增 `tests/test_cli_stream.py`（5 例）锁住 runner 事件契约 + stream_query 消费映射（此前 runner/stream 无测试覆盖）。验收：`pytest -q` = **112 passed**、`ruff check` 全绿、`askanswer --graph -` 与 `askanswer.cli:main` 均正常。

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
