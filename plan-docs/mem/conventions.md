# Conventions

- Nodes return partial state dicts that the reducers merge — never mutate `state` in place. When sorcery decides to retry, write the structured `retry_directive` (dict); when answer consumes it, return `retry_directive: {}` to clear.
- Module-level side effects are forbidden in `schema.py` (no stub graphs); keep it pure data. `persistence.py` must stay lazy — don't import it from `graph.py` at module top.
- New tools should be registered through `registry.py` with the right tag set, not bound directly to the model. Anything needing user approval must set `confirmation_class="shell" | "fs_write"` (more classes go in `_KNOWN_CONFIRM_CLASSES` + a matching `_run_*_with_confirmation`) — not the legacy `requires_confirmation=True` (which is now a read-only compatibility property).
- New file extensions for `read_file` should be added to `FILE_EXTENSIONS` in `intents/base.py` so the file-path regex used by `FileReadHandler.local_classify` (and `IntentRegistry._fallback_classification`) picks them up.
- New intents go in `intents/<name>.py` + a line in `intents/__init__.py`. Lower `priority` runs first; place the global fallback heuristic in `IntentRegistry._fallback_classification`, not in any handler.
- Prefer the typed `ContextSchema` for per-invocation config; only fall back to env vars at the CLI boundary. Inside tools, read it via `ToolRuntime[ContextSchema]` and `normalize_context`.
