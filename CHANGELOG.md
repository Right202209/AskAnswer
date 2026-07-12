# Changelog

## 2026-07-12 · split cli into package (C1)

### Changed
- `askanswer/cli.py` (2510-line single file) is now an `askanswer/cli/` package — one cohesive module per concern, every file ≤300 lines: `theme`/`text` (styling + text-measure primitives), `render`/`progress` (panels + `⏺` node markers), `confirm` (HITL menus), `stream` (`stream_query`), `repl` (`interactive_loop` + `!<cmd>`), `app` (parser / `--graph` export / MCP autoconnect / telemetry init), and `commands/` (router `__init__` + `model`/`mcp`(+`mcp_view`)/`threads`/`timetravel`/`audit`/`transfer` + shared `_common`). `cli/__init__.py` (92 lines) keeps `main` plus back-compat re-exports, so `from askanswer.cli import main/stream_query/handle_command/render_answer/…` and the `askanswer` / `askanswer.cli:main` entry points are unchanged.
- `cli.stream.stream_query` now consumes `runner.stream_leg` directly instead of driving its own `app.stream(["updates","messages"])` loop — closing the C1 goal of a single event-stream implementation shared by CLI and HTTP (the CLI's `_handle_message_chunk` / `_extract_interrupt_value` / `_pending_interrupt` duplicates are gone; their logic already lives in `runner`). Turn-level bookkeeping (audit `begin_run`/`end_run`, telemetry root span, `thread_meta` upsert) stays in the CLI, so multi-leg audit granularity is unchanged. Behavior-preserving for all real (dict) interrupt payloads.

### Added
- `tests/test_cli_stream.py` (5 cases): locks `runner.stream_leg`'s event contract (token/tool/node/interrupt/final ordering, interrupt-without-final, state fallback) and `stream_query`'s consumption mapping (streamed answer, confirmation prompt + resume). This seam was previously untested; it also guards the HTTP server, which shares `runner`.

### Verification
- `pytest -q` = **112 passed** (107 pre-existing + 5 new); `ruff check askanswer tests` clean; `askanswer --graph -` exports nodes; command/render paths smoke-tested against a temp DB. Pure structural move + the documented runner convergence — no user-facing behavior change.

## 2026-07-12 · HTTP/SSE server (C3)

### Added
- `askanswer/runner.py`: UI-free event stream over one graph "leg" — typed `RunEvent`s (`token` / `tool` / `node` / `interrupt` / `final`) mirroring `cli.stream_query` consumption semantics. `run_leg` wraps the stream with the CLI-parity bookkeeping (audit `begin_run`/`flush_pending`/`end_run`, telemetry root span, `thread_meta` upsert) inside a generator `finally`, so client disconnects still get accounted. `runtime_context_from_env()` is now the single env→`ContextSchema` mapping.
- `askanswer/server.py`: stdlib-only `ThreadingHTTPServer` + SSE, zero new dependencies (`python -m askanswer.server`; `askanswer-server` script after reinstall). Endpoints: `GET /health`, `GET /v1/interrupt?thread_id=`, `POST /v1/query`, `POST /v1/resume`. HITL over HTTP: the stream ends at the `interrupt` event with `done.status=interrupted` (pause persisted by the shared checkpointer); `/v1/resume` continues with the same decision shapes the CLI uses. Guards: optional `ASKANSWER_SERVER_TOKEN` bearer auth (`compare_digest`), localhost-only `Origin` + JSON-only content type (CSRF double gate), 64KB body cap, 2-slot run semaphore (503), per-thread busy lock (409), 60s socket timeout, generic error bodies (stacks server-log only).
- `askanswer/wire.py`: transport helpers — SSE framing, `event_wire` (node updates → scalar-only summaries; interrupt payloads → depth-capped `json_safe`), request validation (`RequestError`, `normalize_thread_id`, `split_path`, `is_local_origin`).

### Changed
- `cli._runtime_context` now delegates to `runner.runtime_context_from_env()` — one env→context mapping shared by CLI and HTTP. No other CLI behavior change; `stream_query` still owns its own loop until the C1 split rewires it onto the runner.

### Verification
- **Not yet run** (code-only session). Matrix groups G5+G6+G7 in `docs/important-documentation-verification-matrix.md` gate C3 acceptance; the C3 box in `plan-docs/02-execution-plan.md` stays unchecked until they pass.

## 2026-07-12 · generic clarification protocol (C2)

### Added
- Optional `clarify(state, context)` capability on intent handlers: `ClarificationChoice` / `ClarificationRequest` dataclasses plus the exception-safe `get_clarification` dispatcher in `intents/base.py`. Implemented today by `file_read` (no path → skip / enter one manually), `sql` (no `db_dsn` → proceed as DB question / switch intent to chat), `research` (topic shorter than `RESEARCH_SCOPE_MIN_CHARS=24` → pick a focus angle).
- `clarify_node` (`askanswer/clarify.py`) at the react-subgraph entry (`START → clarify → answer`): runs only on the first answer pass (`step == "understood"`, so sorcery retries and the answer⇄tools loop never re-ask), interrupts with `{"type": "clarify", ...}`, resumes via `Command(resume={"index", "text"})` and merges the chosen updates into `SearchState`. Parent graph topology unchanged; zero-cost pass-through when no handler asks.
- CLI: `_prompt_clarification` arrow-key menu (with optional free-text entry), `⏺ Clarify` progress marker, spinner phase text. Non-TTY takes the default choice, and every default equals "no change" — non-interactive behavior is non-regressive by construction.

### Verification
- **Not yet run** (code-only session). The mandatory matrix lives in `docs/important-documentation-verification-matrix.md`; groups G0+G2+G3+G4 gate C2 acceptance, and the C2 box in `plan-docs/02-execution-plan.md` stays unchecked until they pass.

## 2026-07-11 · audit redaction for paid-API confirmations

### Security
- `external_api_paid` confirmations now redact sensitive values from the audit trail (`args_summary`): keys matching api_key/token/secret/password/authorization/credential are masked, and email addresses inside string values are scrubbed, on both approve and reject paths (`confirmations.redact_audit_args`). The interactive confirmation UI still shows raw args so the user approves what actually runs.

## 2026-07-11 · possibility-space phases 1.2–3.4

### Added
- **MCP health + profile autoconnect** (Phase 1.2): `_ServerEntry` tracks `status`/`last_checked`/`last_error`; `MCPClientManager.health_check()` probes with a 3s timeout. New `mcp_profile.py` persists connections to `~/.askanswer/mcp.json` (atomic writes); `cli._autoconnect_mcp_profile()` replays them on startup. New CLI: `/mcp add_stdio`, `/mcp health`, `/mcp list -v`. `registry.refresh_mcp()` drops `disconnected` servers' tools.
- **Tenant isolation** (Phase 1.3): schema **v3** adds `tenant_id` to `thread_meta`+`audit_event` with per-tenant indexes. All persistence read methods take `tenant_id` (None = unrestricted); writes stamp it (default `ASKANSWER_TENANT_ID`). CLI commands pass `_current_tenant()`; `delete_thread` re-checks ownership. SQL agent connection cache keyed by `(tenant_id, dsn)`. `audit.begin_run` captures tenant per-run.
- **/undo labels** (Phase 2.1): schema **v4** adds `checkpoint_label`. `/undo [n] --label NAME` names a restore point; `/undo --label NAME` resolves back to it; `/checkpoints` shows a label column and `/undo` reports affected message count.
- **Observability exporters** (Phase 2.2): new `telemetry/` package with LangSmith + OpenTelemetry exporters, env-gated (`LANGSMITH_API_KEY` / `ASKANSWER_OTEL_EXPORTER`), zero-overhead when off. Contextvar span stack, LLM callback injected in `load.py`, tool-call events in `_react_internals`, root span in `stream_query`. No `SearchState` fields.
- **research_brief subgraph** (Phase 3.1): `research/` package + `ResearchHandler`; `research_brief_loop` tool runs `plan_queries → search → synthesize → source_check`, calling `tavily_search` via the registry, and returns a cited Markdown brief.
- **decision_memo subgraph** (Phase 3.2): `decision/` package + `DecisionHandler`; `decision_memo_loop` reuses Helix's `interview_node` then a `decide` node → tradeoff memo (`max_retries=0`).
- **Helix as MCP server** (Phase 3.4): `helix_mcp.py` exposes `helix_spec_loop` over stdio via `FastMCP` (`python -m askanswer.helix_mcp`); no `graph.py` import, non-TTY falls back to `default_answer`.

### Verified
- `python -m py_compile` on all changed/new files.

## 2026-07-11

### Added
- Unified HITL confirmation protocol (`confirmations.py`): `shell`, `fs_write`, and `external_api_paid` classes share the same 4-step protocol (plan → gate → interrupt → apply), extensible by adding a handler to `_HANDLERS`.
- `write_file` tool with sensitive-path regex, 1 MB size cap, diff preview, and `fs_write` confirmation class.
- CLI confirmation menus for `fs_write` (diff/preview display) and `external_api_paid` (tool/args/fee warning).

### Changed
- Renamed `shell_plan` graph node to `confirm_plan`; generalized to handle all confirmation classes.
- Generalized `pending_shell` state to `pending_confirmations` (keyed by tool_call_id with class tag); old key preserved for checkpoint compatibility.
- `_react_internals.py`: `_run_with_confirmation` now dispatches via `get_confirmation_handler(clazz)` instead of hardcoded shell logic.
- `timetravel.py`: `CheckpointInfo.pending_shell` → `pending_confirm`; `rewind_to` / `fork_thread` blank both keys.
- `cli.py`: `_prompt_confirmation` dispatcher replaces direct `_prompt_shell_confirmation` call; checkpoint flag → `pending-confirm`; export excludes `pending_confirmations`.

### Verified
- `python -m py_compile` on all changed files.

## 2026-05-06

### Added
- Added an intent handler registry under `askanswer/intents/` with handlers for chat, search, SQL, file reads, and a `math` smoke-test intent.
- Added structured `retry_directive` state for sorcery-driven retries without synthetic user messages.
- Added tag-based tool filtering and `confirmation_class` metadata while preserving `bundle=` compatibility.

### Changed
- Refactored intent classification and answer evaluation so new intents can be registered through handlers instead of editing core graph branches.
- Moved intent-specific ReAct prompt hints and CLI labels into handlers.
- Kept the main graph shape stable: `START -> understand -> answer -> sorcery -> (END | answer)`.

### Verified
- `python -m compileall askanswer`
- `python -m askanswer --graph -`
- Local smoke checks for `math`, `search`, `sql`, `chat`, and `file_read` classification.
