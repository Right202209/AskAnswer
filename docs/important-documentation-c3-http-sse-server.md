# Important Documentation — C3 HTTP/SSE server

> 状态：**代码已落地，尚未运行验证**（本轮遵循「只写代码、不跑 Python」）。
> 本文记录改动、API 契约、设计取舍、安全清单结论；**逐项验证清单与提交闸门已并入**
> `docs/important-documentation-verification-matrix.md`（总表）。
> 对应 `plan-docs/02-execution-plan.md` 的 **C3**；该进度框在总表对应闸门通过前**不勾选**。

## 1. 改动清单

| 文件 | 改动 |
|---|---|
| `askanswer/runner.py`（新，275 行） | UI 无关的"一腿"事件流：`RunEvent`（token/tool/node/interrupt/final）、`stream_leg`（纯事件）、`run_leg`（+审计/telemetry/thread_meta 记账）、`runtime_context_from_env`、`query_input`/`resume_input`/`thread_config`/`preview_of`、`pending_interrupt`/`extract_interrupt_value`/`final_answer_from_state` |
| `askanswer/wire.py`（新，118 行） | 传输层：`sse_frame`（SSE 帧）、`event_wire`（RunEvent→线上载荷）、`json_safe`（递归 JSON 化，深度上限 6）、`RequestError`、`split_path`/`is_local_origin`/`normalize_thread_id` |
| `askanswer/server.py`（新，282 行） | 标准库 `ThreadingHTTPServer` + SSE：4 个端点、鉴权/跨源/体积三道守门、全局并发 + 同 thread 互斥、`python -m askanswer.server` 可启 |
| `askanswer/cli.py` | `_runtime_context` 改为委托 `runner.runtime_context_from_env()`（env→context 单一口径）；新增 import。**stream_query 本体未动** |
| `pyproject.toml` | `[project.scripts]` 增 `askanswer-server = "askanswer.server:main"`（重装后生效；未重装用 `python -m askanswer.server`） |

**未改动**：`graph.py`、`state.py`、`nodes.py`、react 子图、intents、confirmations —— 父图拓扑与 HITL 协议零变化。**零新增依赖**（全部标准库），故无需 pip 下载。

## 2. 与计划的偏差（需知情）

1. 计划写"C3 复用 C1 的 runner"，但 C1（拆 cli.py）被用户跳过未做。本轮**先行落地 C1 里 server 需要的那一块**（`runner.py`），`cli.py` 的 `stream_query` **尚未**改为消费 runner 事件——"CLI 与 HTTP 消费同一事件序列"目前指**同一份消费口径的两个实现**（runner 逐行对照 `stream_query` 语义编写），真正收敛到单实现留给 C1。C1 落地时把 `stream_query` 重写为 `for ev in runner.run_leg(...)` 即可。
2. 计划只提 `server.py` 一个新文件；为守住 ≤300 行/文件的硬指标拆出了 `wire.py`（server 一度 317 行）。`wire.py` 的请求校验部分也是 C4 只读 JSON 端点的复用点。

## 3. HTTP API 契约

- `GET /health` → `{"ok": true}`（免鉴权）。
- `POST /v1/query`，body `{"query": "…", "thread_id": "可选"}` → SSE 流。`thread_id` 缺省服务端生成 uuid4.hex 并在首帧 `meta` 事件返回。
- `POST /v1/resume`，body `{"thread_id": "…", "decision": <resume 值>}` → SSE 续跑。无挂起 interrupt → 409。
- `GET /v1/interrupt?thread_id=…` → `{"thread_id", "interrupt": <载荷|null>}`（断线后恢复确认 UI 用）。

SSE 事件（`event:` 名 + 单行 JSON `data:`）：
`meta`{thread_id} → (`token`{text} | `tool`{names} | `node`{node,summary,elapsed} | `interrupt`{…原载荷})* → (`final`{text})? → `done`{status: completed|interrupted|failed, thread_id}。另有 `error`{message:"internal error"}。

**消费者契约**：收到 `tool` 事件应清空已累计的 token 缓冲（与 CLI `in_tool` 语义一致——规划期文本不属于最终答案）；`interrupt` 后流以 `done.status=interrupted` 结束，用 `payload["type"]` 选确认 UI，`decision` 形态与 CLI `Command(resume=...)` 值完全一致：clarify→`{"index", "text"}`；shell/fs_write→`{"approve": bool, …}`（与 `_prompt_*_confirmation` 的返回一致）。

错误响应统一 `{"error": {"code", "message"}}`：400（参数/JSON 非法）、401（token 不符）、403（跨源）、404、409（thread 忙 / 无挂起 interrupt）、411/413/415（体积与类型守门）、500（细节只进服务端日志）、503（并发槽满）。

## 4. 设计不变量

1. **同一张图**：server 懒加载一次 `create_search_assistant()`（默认共享 SqliteSaver checkpointer），多请求线程共享；`import askanswer.server` 不触发 graph/persistence（`_get_app` 内 lazy import）。
2. **HITL 协议不变**（spec 不变量 7）：HTTP 只是把 CLI 的"内联询问"换成"断流 + 后续 /v1/resume"；副作用工具体仍只在 confirmation resume 后执行。
3. **记账口径与 CLI finally 一致**：`run_leg` 的 finally 做 flush_pending + close_span + end_run + upsert_meta；audit 用 ContextVar，请求线程天然隔离；客户端断连（generator close / GeneratorExit）也会走 finally，记账不丢。resume 腿 `preview=None`，靠 `upsert_meta` 的 COALESCE 不覆盖旧 preview。
4. **并发安全**：全局 `BoundedSemaphore(2)` 限流 + `_BUSY_THREADS` 集合保证同 thread_id 串行（并发同 thread → 409）；SqliteSaver 单连接由 persistence 的锁与 `check_same_thread=False` 保护。
5. **tenant 口径**：server 进程级（`ASKANSWER_TENANT_ID` env，经 `runtime_context_from_env`），与 CLI 相同；未做每请求租户头。

## 5. 安全清单结论（rules/security.md 逐项）

- 硬编码密钥：无；Bearer token 仅从 `ASKANSWER_SERVER_TOKEN` env 读，比较用 `hmac.compare_digest`（防时序侧信道）。
- 输入验证：query 1..8000 字符；thread_id 白名单正则 `^[A-Za-z0-9._-]{1,64}$`；body ≤64KB 且必须 `application/json` + Content-Length；decision 原样进 `Command(resume=...)`（下游 confirmations 已有 gate/danger 二次校验）。
- SQL 注入：server 层无 SQL；经 persistence 的参数化查询。
- XSS：只回 JSON/SSE，不渲染 HTML（C4 做 UI 时另查）。
- CSRF：Origin 非 localhost 一律 403 + 强制 JSON Content-Type（HTML form 无法伪造）双闸。
- 认证/授权：默认只绑 127.0.0.1；绑非本机地址且未设 token 时启动打显式警告。
- 限流：并发槽（503）+ 同 thread 互斥（409）+ 体积上限（413）+ socket 超时 60s（防 Slowloris）。
- 错误不泄密：客户端只见静态文案与 "internal error"；栈/异常细节仅进服务端日志。已知暴露面：interrupt 载荷含确认详情（shell 命令、文件 diff）——这本就是给确认者看的，受鉴权/本机绑定保护。

## 6. 必测清单

> **已迁移**：本节原逐项清单已并入总表
> `docs/important-documentation-verification-matrix.md`（条目编号 R2/R3 + C3-U1..U9 + C3-E1..E8 + C3-R1..R2），
> 执行与勾选一律以总表为准，本节不再维护逐项框。原 6.1 冒烟 → 总表 G0 + C3-E1/C3-R1；6.2 单元 → G5；6.3 端到端 → G6；6.4 非回归 → G7。

## 7. 已知取舍 / 待办
- CLI 与 runner 的事件消费逻辑暂时双实现（见 §2），C1 收敛；在那之前改流式语义要**两处同步改**。
- `MAX_CONCURRENT_RUNS=2`、`SOCKET_TIMEOUT_SECONDS=60`、`MAX_QUERY_CHARS=8000` 为保守初值，按实测调。
- 客户端断连只能在下一个事件边界停止该腿（LangGraph 流被 close 后不再推进）；无显式 cancel 端点。
- resume 前的 pending_interrupt 预检查存在良性 TOCTOU（竞态时 LangGraph 自身报错 → SSE error 事件收尾）。
- 每请求租户头、CORS 白名单开放、图状态只读端点（threads/messages/audit）留给 C4。
- 提交前仍需：更新 `CHANGELOG.md`；通过总表矩阵的 G5+G6+G7（且 C2 已先行提交）后再勾选 `02-execution-plan.md` 的 C3（提交信息 `feat: http sse server`；cli.py 按 C2/C3 拆 hunk 的步骤见总表 §1/§4）。
