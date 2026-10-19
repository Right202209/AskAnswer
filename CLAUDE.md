# CLAUDE.md

Guidance for Claude Code (and other coding agents) working in this repository.
This file is self-contained: everything needed to work on the code is here, and
nothing depends on gitignored files. Architecture rationale and the refactoring
history live in `docs/design/ARCH_REVIEW_CORE_AGENT.md` and `CHANGELOG.md`.

## What this is

`askanswer` is an **embeddable LangGraph agent core** (a library, no CLI/server).
Hosts compile a graph with `create_search_assistant()`, drive it with
`run_leg()` / `stream_leg()`, and handle `interrupt` events (clarification and
human approval) by resuming with `resume_input(decision)`.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"          # add ,telemetry for LangSmith/OTEL
python -m pytest                           # full suite, no API key or network needed
python -m pytest tests/core/test_runner.py -k inmemory   # single test
python -m pytest -W error                  # what CI runs (warnings are errors)
python -m ruff check askanswer tests       # lint (E, F, I); --fix for import order
python scripts/import_smoke.py             # run from an empty dir: import has no side effects
```

Test isolation (`tests/conftest.py`): env vars that change behaviour are cleared
per test, the session opts out of settings/.env bootstrap
(`bootstrap_environ(enabled=False)`), `pm` gives a temp-file `PersistenceManager`,
and `fake_llm` swaps the global model proxy for a scripted fake chat model.

## Package layout

| Package | Responsibility |
| --- | --- |
| `askanswer/__init__.py` | Public facade — exports must stay exactly: `ContextSchema`, `RunEvent`, `create_search_assistant`, `normalize_context`, `query_input`, `resume_input`, `run_leg`, `stream_leg`, `thread_config` |
| `core/` | `graph.py` (parent graph), `nodes.py` (understand / sorcery), `runner.py` (events + bookkeeping), `state.py` (`SearchState`), `context.py` (token budget), `clarify.py`, `confirmations.py` (HITL handlers), `react/` (`__init__` builds the subgraph, `answering.py` = answer node, `dispatch.py` = tools / confirm nodes) |
| `config/` | `settings.py` (layered settings + lazy bootstrap, `resolve_db_dsn`), `schema.py` (`ContextSchema`) |
| `llm/` | `proxy.py` (hot-swappable global `model`, callback injection), `routing.py` (`model_for(role)`, fallbacks) |
| `intents/` | `IntentHandler` protocol (`base.py`) + 8 handlers + `IntentRegistry` |
| `tools/` | `registry.py` (`ToolRegistry`, seeding), `builtin.py`, `filesystem.py`, `shell.py`, `mcp_manager.py`, `mcp_adapter.py` |
| `subagents/` | `sql/` (native SQLAlchemy), `helix/`, `research/`, `decision/`, `common/interview.py` |
| `storage/` | `manager.py` (SQLite, `SqliteSaver`, singleton, `NullPersistence`), `migrations.py` (schema v1–v5), `threads.py`, `audit_store.py` |
| `observability/` | `audit.py` (contextvar buffer), `redaction.py`, `telemetry.py`, `langsmith.py`, `otel.py` |

## Architecture invariants (do not break)

1. **Parent graph never forks by intent**: `START → understand → answer → sorcery → (END | answer)`.
   Intents parameterise behaviour through `IntentHandler` (classification, prompt hint,
   `bundle_tags`, `evaluate`, optional `clarify`). Add an intent with
   `askanswer.intents.register(handler)`, never with new parent-graph nodes.
2. **Tools only via the registry** (`askanswer.tools.get_registry()`); tags decide which
   intents see a tool. Plain tool calls run through `ToolNode` so `ToolRuntime[ContextSchema]`
   is injected; never call `tool.invoke` directly for tools that may need the runtime.
3. **Four-step HITL protocol** for tools with a `confirmation_class`
   (`shell` / `fs_write` / `external_api_paid`): `plan()` → `gate()` → `interrupt_payload()`
   → `apply()`. The plan is persisted in `state["pending_confirmations"]` *before*
   `interrupt()` and reused on resume (no LLM re-run); `apply()` re-checks safety gates
   (the user may have edited the command). New classes go in `core/confirmations.py::_HANDLERS`,
   not as if/elif in `dispatch.py`.
4. **Nodes return partial state dicts**; `messages` uses the `add_messages` reducer.
   Observability (audit buffer, telemetry spans) lives in `contextvars`, never in `SearchState`.
5. **No import-time side effects**: importing `askanswer` must not modify `os.environ`, read
   config files, open SQLite, build a model or need an API key. Read env vars at call time.
   Configuration is loaded once by `config.settings.ensure_bootstrapped()` from the entry
   points (`create_search_assistant`, model construction / default-label resolution,
   `model_for`, `get_persistence`, `runtime_context_from_env`, telemetry init, Tavily /
   OpenWeather getters). `bootstrap_environ(enabled=False)` opts out.
6. **Persistence follows the checkpointer**: `run_leg()` writes audit + `thread_meta` to the
   `PersistenceManager` owning `app.checkpointer`, to `persistence=` if given, otherwise to
   `NullPersistence` (in-memory runs never create `state.db`).
7. **Zero-cost disabled paths**: unset `ASKANSWER_CONTEXT_MAX_TOKENS` → no token estimation;
   telemetry disabled → no `langsmith` / `opentelemetry` import.
8. **Prompt-cache-friendly prompts**: stable preamble and tool list first, dynamic context last
   (`core/react/answering.py`); context trimming keeps tool-call / `ToolMessage` blocks atomic.
9. **Security boundaries**: sensitive-path regex for read/write, size limits, dangerous shell
   pattern gate (checked twice), SQL agent is read-only (`_check_sql_readonly` before
   `sql_db_query`), audit args are redacted, SQL connection caches are keyed by
   `(tenant_id, dsn)`, MCP tools default to the `chat` intent only.

## Configuration

Precedence (high → low): process env > `.askanswer/settings.local.json` >
`.askanswer/settings.json` > `~/.askanswer/settings.json` (or `ASKANSWER_SETTINGS` /
`XDG_CONFIG_HOME`) > `.env` (found by walking up from the cwd). Templates:
`.env.example`, `settings.example.json`. Key variables: `OPENAI_API_KEY`,
`ASKANSWER_DEFAULT_MODEL`, `ASKANSWER_MODEL_<ROLE>` / `ASKANSWER_MODEL_FALLBACKS_<ROLE>`,
`TAVILY_API_KEY`, `OPENWEATHER_API_KEY`, `ASKANSWER_DB_DSN` (legacy
`WLANGGRAPH_POSTGRES_DSN` is a deprecated fallback), `ASKANSWER_DB_PATH`,
`ASKANSWER_TENANT_ID`, `ASKANSWER_CONTEXT_MAX_TOKENS`, `ASKANSWER_CONTEXT_DIGEST`,
`ASKANSWER_RUN_TOKEN_BUDGET`, `ASKANSWER_MCP_ALL_INTENTS`, `LANGSMITH_API_KEY`,
`ASKANSWER_OTEL_EXPORTER`.

## Conventions

- Keep `askanswer/__init__.py` exports stable; everything else may move before 1.0 but
  record breaking changes in `CHANGELOG.md`.
- Storage schema changes: bump `storage/migrations.py::SCHEMA_VERSION`, append an
  idempotent `if current < N:` branch, add a migration test in `tests/storage/`.
- Tests mirror the package (`tests/<package>/`); tests must not need network, API keys
  or a real database (use temp SQLite files and the `fake_llm` fixture).
- Inline comments are often Chinese; public module / API docstrings are English or bilingual.
- Do not commit `.env`, `.askanswer/settings.local.json`, `*.egg-info`, `__pycache__`.
