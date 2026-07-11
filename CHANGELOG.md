# Changelog

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
