# Changelog

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
