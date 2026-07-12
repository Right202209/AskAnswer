# Important Documentation — 未提交改动的强制验证矩阵（总表）

> **角色**：工作区所有「未提交（unsubmitted）+ 未运行验证（unverified）」改动的**唯一**执行清单与提交闸门。
> 逐项勾选发生在**本文件**；`docs/important-documentation-c2-clarification.md` 与 `…-c3-http-sse-server.md`
> 保留背景/设计不变量/API 契约，其原清单章节已改为指向本表（避免双源漂移）。
> 全程遵循：`source .venv/bin/activate` 后再执行任何 `python`/`pytest`/`askanswer`。

## 1. 未提交改动全景与提交归属

| 文件 | 提交前状态 | 归属 | 说明 |
|---|---|---|---|
| `askanswer/intents/base.py` | M | **C2** | ClarificationChoice/Request + `get_clarification` |
| `askanswer/intents/file_read.py` | M | **C2** | `clarify`：缺路径 |
| `askanswer/intents/sql.py` | M | **C2** | `clarify`：缺 DSN |
| `askanswer/intents/research.py` | M | **C2** | `clarify`：范围过宽（`RESEARCH_SCOPE_MIN_CHARS=24`） |
| `askanswer/clarify.py` | ?? | **C2** | react 子图入口节点 `clarify_node` |
| `askanswer/react.py` | M | **C2** | 子图入口 `START → clarify → answer` |
| `askanswer/cli.py` | M | **C2×7 + C3×2**（拆 hunk） | 见下方拆分说明 |
| `askanswer/runner.py` | ?? | **C3** | UI 无关事件流 + 记账 |
| `askanswer/wire.py` | ?? | **C3** | SSE 帧 / json_safe / 请求校验 |
| `askanswer/server.py` | ?? | **C3** | 标准库 HTTP/SSE server |
| `pyproject.toml` | M | **C3** | `askanswer-server` 入口（+2 行） |
| `CLAUDE.md` | M | **C3** | 索引 hook 行更新（1 行） |
| `docs/important-documentation-c2-clarification.md` | ?? | **C2** | 背景文档，随 C2 提交 |
| `docs/important-documentation-c3-http-sse-server.md` | ?? | **C3** | 背景文档，随 C3 提交 |
| `docs/important-documentation-verification-matrix.md` | ?? | 首个落地的 feature 提交携带 | 本文件 |
| `tests/test_{confirmations,graph,intents,mcp_profile,persistence,registry,telemetry}.py` | ??×7 | **B2**（用户暂停中） | 已写、从未运行；`tests/conftest.py` 已随 B1 提交 |
| `.claude/mem/*.md` | gitignored | 本地 | 无需提交、无需运行验证 |

**cli.py 拆 hunk（提交时必须执行）**：C3 仅两个 hunk —— ① import 区新增 `from .runner import runtime_context_from_env`（约 L58，diff 头 `@@ -55,6 +55,7`）；② `_runtime_context` 函数体改为委托（约 L469，diff 头 `@@ -465,12 +467,8`）。其余 7 个 hunk（ui_select import、`_PHASE_TEXT`、`_on_node_update`、`_render_node_update`、`_clarify_detail`、`_prompt_confirmation` 分发、`_prompt_clarification`）全部属于 C2。终端里用 `git add -p askanswer/cli.py` 按上述归属交互式staging。

**提交顺序（硬约束）**：`C2 → C3 → B2`。原因：(a) cli.py 的 runner import 必须与 `runner.py` 同一提交入库，否则 C2 提交若整文件带入 cli.py 会得到 `ImportError` 的坏中间树；(b) B2 测试写于 C2 之前，需先按 §3-G1 分诊/补充后再绿灯提交。

## 2. 执行前置

- `source .venv/bin/activate`；确认 `pytest --version`、`ruff --version` 可用（B1 已提交 dev extras；若缺：`pip install -e ".[dev]"`）。
- **隔离数据库**：所有验证跑在 `export ASKANSWER_DB_PATH=$(mktemp -d)/t.db` 下，绝不碰 `~/.askanswer/state.db`。
- **API key 需求**：G0/G1/G2/G3/G5 全部免 key（mock/纯逻辑）；G4/G6/G7 里标 🔑 的项需要 `OPENAI_API_KEY`（真实回答），其余用 mock LLM（monkeypatch `load.model`）即可。
- 非 TTY 项用管道模拟：`echo "问题" | askanswer` 或 pytest 内 subprocess。

## 3. 验证矩阵（按组顺序执行，组内任一失败按 §5 处理）

### G0 · 全局冒烟（免 key，先跑）
- [ ] **R1** `python -c "from askanswer.graph import create_search_assistant"` 导入成功（无循环 import：react → clarify → intents/schema/state）。
- [ ] **R2** `python -c "import askanswer.server, askanswer.runner, askanswer.wire"` 成功，且当前目录/ASKANSWER_DB_PATH 下**未创建** state.db（server 的 `_get_app` 懒加载生效）。
- [ ] **R3** `askanswer --graph -` 含 `understand/answer/sorcery/confirm_plan` 且与改动前一致；`clarify` 节点验证：若 `--graph` 渲染子图内部（xray）则应含 `clarify`，否则改用 `python -c "from askanswer.react import build_react_subgraph; print('clarify' in build_react_subgraph().get_graph().nodes)"` → True。
- [ ] **R4** `python -c "import askanswer.helix_mcp"` 不触发 persistence/graph（隔离未被 C2/C3 破坏）。
- [ ] **R5** intent 注册表仍为 8 个 handler（file_read, sql, helix, decision, math, research, search, chat）。

### G1 · B2 既有测试套件（已写未运行；与新改动的交叉已预判）
- [ ] **T1** `pytest -q` 全绿、≥25 用例（7 个文件 + conftest fixture 生效：临时 DB、清 TENANT/LANGSMITH/OTEL env）。
- [ ] **T2** 交叉预判确认：`test_graph.py` 的 `{"answer","tools","confirm_plan"} <= nodes` 是子集断言，clarify 新增**不应**使其失败；`test_intents.py` 无 clarify 假设。若两者失败 = 真回归，不是测试过时。
- [ ] **T3** 失败分诊：先判「测试过时（写于 C2 前）」vs「代码回归」；过时 → 修测试（随 B2 的 `test:` 提交），回归 → 停下修代码并重跑所属组。

### G2 · C2 单元（mock LLM/网络；建议落到 `tests/test_clarify.py` + 扩展 `tests/test_intents.py`）
- [ ] **C2-U1** `FileReadHandler().clarify({"file_path":""}, ctx)` 返回 request；`{"file_path":"/a.txt"}` 返回 None。
- [ ] **C2-U2** `SqlHandler().clarify(state, ContextSchema(db_dsn=None))` 返回 request 且第 2 项 `updates=={"intent":"chat"}`；`db_dsn="postg://…"` 返回 None。
- [ ] **C2-U3** `ResearchHandler().clarify({"user_query":"RAG"}, ctx)` 返回 request；≥24 字查询返回 None；空查询返回 None。
- [ ] **C2-U4** `get_clarification` 对无 `clarify` 的 handler（chat/math/helix/decision/search）返回 None；对抛异常的 handler 返回 None。
- [ ] **C2-U5** `clarify._resolve`：普通项→对应 `updates`；手动输入项（index==len(choices)）+非空 text→`{field:text}`；空 text→`{}`；index=-1→`{}`。
- [ ] **C2-U6** `clarify._read_decision`：dict / 非 dict / 非法 index 的兼容（均不抛异常）。

### G3 · C2 图/节点（mock LLM，`InMemorySaver`，走 `stream` + `Command(resume=...)`）
- [ ] **C2-G1** `clarify_node` 在 `step!="understood"` 时返回 `{}`（sorcery 重试轮与 answer⇄tools 内循环不重入澄清）。
- [ ] **C2-G2** 首轮 file_read 无路径：stream 收到 `__interrupt__`（`type=="clarify"`）；`Command(resume={"index":1,"text":"/tmp/x.txt"})` 后 `state["file_path"]=="/tmp/x.txt"`。
- [ ] **C2-G3** 首轮 sql 无 DSN：resume `{"index":1}` 后 `state["intent"]=="chat"`，answer 绑定 chat 工具集（无 `sql_query`）。
- [ ] **C2-G4** 首轮 research 短主题：resume `{"index":2}` 后 `state["user_query"]` 含「侧重技术原理…」。
- [ ] **C2-G5** **不重跑 LLM**：断言 understand 的 `_intent_from_llm` 在 resume 前后调用次数不变。

### G4 · C2 非回归
- [ ] **C2-R1** 非 TTY（管道/CI）跑三场景：各取默认项，行为与接入前一致（file_read 不读文件、sql 仍按库、research 广泛）。🔑（或 mock）
- [ ] **C2-R2** chat/math/search 等无 clarify 的 intent：`clarify_node` 返回 `{}`，不打印 `⏺ Clarify` 标记，spinner 不闪「澄清需求…」。🔑（或 mock）
- [ ] **C2-R3** 一轮内「先 clarify 后 fs_write/shell 确认」两次 interrupt 顺序 resume 正常（CLI `stream_query` while 循环逐个处理）。
- [ ] **C2-R4** 旧 checkpoint（无 clarify 节点历史的库）恢复对话不报错。

### G5 · C3 单元（mock；建议 `tests/test_runner.py` / `tests/test_wire.py`）
- [ ] **C3-U1** `wire.sse_frame("token", {"text":"a\nb"})`：单行 data（换行被 JSON 转义）+ `\n\n` 结尾。
- [ ] **C3-U2** `wire.json_safe`：嵌套 dict/list 原样；对象→str；深度 >6 截断为 str；set→list。
- [ ] **C3-U3** `wire.normalize_thread_id`：空→None；`"a/b"` 与 65 字符→`RequestError(400)`；合法值原样。
- [ ] **C3-U4** `wire.event_wire`：五类事件载荷形状正确；node 摘要只含标量、字符串截到 200。
- [ ] **C3-U5** `runner._message_events`：含 `tool_call_chunks` 的 AIMessageChunk → 同一规划阶段恰发一次 `tool` 事件；纯 content → `token`；非 AIMessageChunk / 空 content → 无事件。
- [ ] **C3-U6** `runner._update_events`：`__interrupt__` 不产 node 事件且载荷被记录；answer/sorcery 的 `final_answer` 被捕获。
- [ ] **C3-U7** `runner.stream_leg`（fake app）：无 interrupt → 尾事件 `final`；有 → 尾事件 `interrupt`；流结束无 `__interrupt__` 但 `get_state().tasks` 挂起 → 仍出 `interrupt`（兜底路径）。
- [ ] **C3-U8** `runner.run_leg`：正常结束/中途 `close()` 都恰好一次 flush_pending + end_run + upsert_meta（mock audit/persistence 断言）；resume 腿 preview=None（COALESCE 保旧值）。
- [ ] **C3-U9** `runner.preview_of`：折叠空白、80 截断、空→None。

### G6 · C3 端到端（mock LLM + 真实 server 线程；SSE 客户端逐事件断言）
- [ ] **C3-E1** `python -m askanswer.server --port 8765` 启动；`curl -s localhost:8765/health` → `{"ok": true}`。
- [ ] **C3-E2** /v1/query 简单问答：事件序列 `meta → …token… → node(answer) → node(sorcery) → final → done{completed}`；同输入下答案与 CLI 一致。🔑（或 mock）
- [ ] **C3-E3** /v1/query 触发 clarify（首轮 file_read 无路径）：`interrupt{type=clarify} → done{interrupted}`；`GET /v1/interrupt` 取回同一载荷；`/v1/resume {"decision":{"index":1,"text":"/tmp/x.txt"}}` 续流至 final 且 `state.file_path` 生效。
- [ ] **C3-E4** /v1/query 触发 shell 确认：`interrupt{type=confirm_shell}` → resume `{"approve": false}` → 优雅收尾且**工具体未执行**（spec 不变量 7）。
- [ ] **C3-E5** 并发闸：同 thread 第二个 /v1/query → 409；并发槽占满 → 503；无挂起 interrupt 的 /v1/resume → 409。
- [ ] **C3-E6** 守门：设 `ASKANSWER_SERVER_TOKEN` 后无/错 token → 401 而 `/health` 仍 200；`Origin: http://evil.test` → 403；`Content-Type: text/plain` → 415；body >64KB → 413；无 Content-Length → 411；非法 JSON → 400。
- [ ] **C3-E7** 客户端中途断开：服务端日志 "client disconnected"；audit 行仍落库；同 thread 立即可再 /v1/query（busy 槽已释放）。
- [ ] **C3-E8** 记账口径：HTTP 一轮后 CLI `/audit` 可见该 thread 事件、tenant 过滤与 CLI 一致；thread_meta 的 preview/message_count 正确，resume 腿未覆盖 preview。

### G7 · C3 非回归
- [ ] **C3-R1** CLI 三场景（chat / file_read+clarify / shell 确认）行为与改动前一致 —— cli.py 的 C3 改动仅 `_runtime_context` 委托，重点确认 SQL intent 仍能拿到 `WLANGGRAPH_POSTGRES_DSN`。🔑（或 mock）
- [ ] **C3-R2** `pip install -e .` 后 `askanswer-server` 入口可用（等价 `python -m askanswer.server`）。

## 4. 提交闸门（全过对应组 + 附加动作后才可 commit）

| 顺序 | commit message | 携带文件 | 前置组 | 附加动作 |
|---|---|---|---|---|
| 1 | `feat: generic clarification protocol` | intents/4 文件、clarify.py、react.py、cli.py（**仅 C2 的 7 个 hunk**）、docs/…-c2-…md、本文件 | G0 + G2 + G3 + G4 | ① rules/security.md 逐项清单；② `CHANGELOG.md` 记 C2；③ 勾选 `plan-docs/02-execution-plan.md` 的 C2 框 |
| 2 | `feat: http sse server` | runner.py、wire.py、server.py、cli.py（**剩余 2 个 C3 hunk**）、pyproject.toml、CLAUDE.md、docs/…-c3-…md | G5 + G6 + G7（且 C2 已提交） | 同上三步（CHANGELOG 记 C3；勾 C3 框） |
| 3 | `test: cover persistence, confirmations, mcp profile, telemetry, registry` | tests/ 7 文件（+按 G2/G3/G5 新增的 test_clarify/test_runner/test_wire 等） | G1（用户解除 B2 暂停后） | 勾选 B2 框；按 B2 验收「≥25 用例」复核 |

通用提交规则：`<type>: <description>` 格式、**无 Co-Authored-By**；每次提交前重跑该批次所属组中标 R 的非回归项。

> **2026-07-12 状态**：应用户指示，提交 1–2（C2/C3）在验证组执行前**先行落库** —— 代码入库 ≠ 验收通过。
> §3 各组仍全部未执行；`02-execution-plan.md` 的 C2/C3 进度框保持未勾，待对应闸门组全绿后补勾。
> 提交 3（B2 测试）随暂停状态继续搁置（7 个 test 文件仍为未跟踪）。

## 5. 失败处理协议

1. **代码回归**：停止推进 → 修复 → 重跑**整组**（不是只重跑失败项）。
2. **安全发现**：按 rules/security.md 响应协议 —— STOP、全面安全审查、CRITICAL 先修、必要时轮换密钥、全库排查同类。
3. **测试过时**（仅 G1 可能）：修测试而非代码，归入提交 3。
4. 任何组通过后立即在本文件勾选并注明日期，避免重复劳动。

## 6. 维护规则

- 新产生的未提交改动必须同步登记：§1 表格 + §3 新增分组/条目；否则视为未覆盖。
- 勾选只发生在本文件；`02-execution-plan.md` 的进度框只在 §4 对应闸门整体通过后勾选。
- C2/C3 背景文档中的原清单章节已指向本表，**不要**在那两处恢复逐项框。
