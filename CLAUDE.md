# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install (editable): `pip install -e .` (Python 3.10+)
- Run CLI: `askanswer "<question>"` (single-shot) · `askanswer` (REPL) · `python -m askanswer`
- Export the LangGraph topology: `askanswer --graph` (stdout) or `askanswer --graph docs/graph.mmd`
- No test suite or linter is wired up yet; do not invent commands for them. The repo is verified by `python -m compileall askanswer` + manual smoke checks documented in `CHANGELOG.md`.

## Required env (`.env` at repo root)

`OPENAI_API_KEY`, `TAVILY_API_KEY` are required. Optional: `OPENWEATHER_API_KEY` (weather tool), `WLANGGRAPH_POSTGRES_DSN` + `ASKANSWER_DB_DIALECT` (SQL agent default DB), `ASKANSWER_TENANT_ID`, `ASKANSWER_DB_PATH` / `XDG_DATA_HOME` (override the SQLite state file location, defaults to `~/.askanswer/state.db`).

## Architecture

### Main graph (`askanswer/graph.py`)

Single-entry `create_search_assistant(checkpointer=None)` returns a compiled `StateGraph(SearchState, context_schema=ContextSchema)`. When `checkpointer is None` it lazy-imports `persistence.get_persistence().checkpointer` (a shared `SqliteSaver`); pass `InMemorySaver()` to bypass disk (used by `draw_search_assistant_mermaid` so `--graph` doesn't create an empty `state.db`). Topology is **intent-agnostic at the parent level**:

```
START → understand → answer → sorcery → (END | answer for retry)
```

All five intents (`file_read | sql | chat | search | math`, plus any plugin-registered ones) flow through the same `answer` node (the compiled react subgraph). Intent only affects (a) the system-prompt hint produced by the active `IntentHandler.prompt_hint(state)`, and (b) which tool **tag set** the registry exposes (see Tool registry). `tavily_search` and `read_file` are registered tools, not parent-graph nodes — the LLM decides when to call them. `route_from_sorcery` is the only conditional edge: when sorcery sets `step="retry_search"` it re-enters `answer`; the answer node consumes the structured `retry_directive` field (set by sorcery) and bakes it into the system prompt — there is no longer a synthetic `HumanMessage` injection.

### State (`askanswer/state.py`)

`SearchState` is a `TypedDict` with `messages` (LangGraph `add_messages` reducer), plus flat fields: `user_query`, `search_query`, `search_results`, `final_answer`, `intent`, `file_path`, `retry_count`, `step`, `retry_directive` (sorcery → answer hand-off, cleared after consumption), `pending_shell` (HITL persistence). **When adding a new node that needs to persist data, add the field here** — nodes return partial dicts that get merged.

### Runtime context (`askanswer/schema.py`)

`ContextSchema` (dataclass: `llm_provider`, `db_dsn`, `db_dialect`, `tenant_id`) is the LangGraph **runtime context**, distinct from `SearchState`. The CLI builds it from env vars in `cli.py:_runtime_context()` and passes it via `app.stream(..., context=...)`. SQL nodes accept it as their second arg via `Runtime[ContextSchema]` and read it with `normalize_context(getattr(runtime, "context", None))` (which also accepts plain dicts). Use this for per-invocation config that shouldn't pollute the message-merging state.

### Intent handlers (`askanswer/intents/`)

Each intent is an `IntentHandler` (Protocol in `intents/base.py`) with: `name`, `priority` (lower wins), `bundle_tags` (frozenset of registry tags it wants the LLM to see), `max_retries`, plus four methods — `local_classify(text) → IntentClassification | None`, `prompt_hint(state) → str`, `evaluate(state) → EvaluationResult`, `cli_label(update) → str`. `IntentRegistry` (singleton via `get_intent_registry()`) owns:

- Iteration order = priority order; the first handler whose `local_classify` returns non-None wins.
- The **cross-intent fallback** (long text without `?` → search; chat starters → chat; empty → chat) — handlers don't own global heuristics.
- Field normalization (`IntentClassification` is a Pydantic model with a `model_validator(mode="after")`).
- The `llm_intent_list()` helper used to template the LLM-classifier prompt.

`understand_query_node` runs `classify_local()` first; only if it returns None does it call the LLM (`_intent_from_llm` uses `model.with_structured_output(IntentClassification)`); LLM failures fall back to `classify_local(..., fallback=True)`. The same `classify_local()` is also used inside the react loop by `_reclassify_intent` to handle mid-conversation topic shifts (chat → SQL etc.); it deliberately skips synthetic retry messages so retries don't flip intent.

**To add an intent**: drop a `<name>.py` under `intents/` exporting a class that satisfies the protocol, then register it in `intents/__init__.py`. The registry already routes unknown intent strings to `search` as a safe default. `MathHandler` (`intents/math.py`) is the smoke-test reference — it shows the minimal shape (no retry, no extra tools, just `prompt_hint` + classifier).

### Parent-graph nodes (`askanswer/nodes.py`)

Two node bodies live here: `understand_query_node` (the first node) and `sorcery_answer_node` (the third). `understand_query_node` runs `_local_intent` → `_intent_from_llm` (structured output via `model.with_structured_output(IntentClassification)`) → `_local_intent(fallback=True)`, then writes `user_query`/`search_query`/`file_path`/`intent`/`retry_count=0`/`retry_directive={}`/`step="understood"` plus a one-line AIMessage hint. `sorcery_answer_node` looks up the active handler, returns `_finalize(state)` once `retry_count >= handler.max_retries`, otherwise calls `handler.evaluate(state)` and either finalizes (`decision == "pass"`) or writes `step="retry_search"` + `retry_directive` for the next answer pass. Evaluation logic itself does **not** live here — it's in each `IntentHandler.evaluate`.

### React subgraph (`askanswer/react.py`, `askanswer/_react_internals.py`)

`build_react_subgraph()` compiles the answer ⇄ tools / shell_plan loop and is wired into the parent graph as the `answer` node. It shares `SearchState` with the parent and is compiled **without its own checkpointer** so `interrupt()` from the shell HITL flow propagates to the parent stream. `_react_internals.py` holds the actual node bodies (`_answer_node`, `_shell_plan_node`, `_tools_node`, `_route_from_answer`, `_reclassify_intent`); treat it as private — call through `react.build_react_subgraph()`.

`_answer_node` (a) runs `_reclassify_intent` and writes the new intent back to `state["intent"]` if it changed, (b) calls `handler.prompt_hint(state)` for the system-prompt prefix, (c) appends `retry_directive["instruction"]` if present and clears the field on its next return, and (d) binds only `registry.list(tags=handler.bundle_tags)`.

`_tools_node` splits tool calls three ways: `plain_calls` go to `langgraph.prebuilt.ToolNode` (which auto-injects `ToolRuntime[ContextSchema]`); `shell_names` (confirmation_class == "shell") go through `_run_with_confirmation` which raises `interrupt()`; any other `confirmation_class` value lands in `unsupported_confirm_calls` and gets a `ToolMessage` explaining the executor isn't wired — the LLM can react instead of crashing. This leaves the door open for `fs_write` / `external_api_paid` classes without forcing a HITL implementation upfront.

### Tool registry (`askanswer/registry.py`)

Process-wide `ToolRegistry` (`get_registry()`) is the single source of truth. Each `ToolDescriptor` carries a `frozenset` of **tags** (intent names are tags, plus free-form labels like `builtin`/`io_bound`/`shell`/`mcp`/`external_api`/`sql_tool`). `_answer_node` filters with `registry.list(tags=handler.bundle_tags)`; `registry.list(bundle="sql")` and the `descriptor.bundles` / `descriptor.requires_confirmation` properties remain as backward-compat aliases. `_BUILTIN_TAGS = ALL_INTENT_TAGS | {builtin, io_bound}`, so adding a new intent automatically gets `tavily_search` / `read_file` / `calculate` / `check_weather` / etc. without touching `_seed_*` registrations. Shell deliberately omits the `sql` tag; the SQL natural-language tool only carries `chat | sql | sql_tool`.

Confirmation: `ToolDescriptor.confirmation_class: Literal["none","shell","fs_write","external_api_paid"]`. `registry.confirmation_classes()` returns `name → class` for the routing in `_route_from_answer`, `_shell_plan_node`, and `_tools_node`. Currently only `gen_shell_commands_run` is `"shell"`. After connecting/disconnecting an MCP server, call `registry.refresh_mcp()` to rebuild the `mcp:*` slice.

### SQL agent (`askanswer/sqlagent/`)

The SQL flow is **exposed as a tool**, not as a parent-graph node. `sql_tool.sql_query` is the registry entry (tags: `chat | sql | sql_tool`); when invoked, it calls `sql_agent.run_sql_agent` which executes a separate `StateGraph(SqlAgentState, context_schema=ContextSchema)` from `build_sql_agent()`. Internal flow: `list_tables → call_get_schema → get_schema → generate_query → (check_query → run_query → generate_query)*`. Hard caps: `MAX_SQL_QUERY_CALLS = 2`, `SQL_RECURSION_LIMIT = 12` — when exceeded, routes to `limit_exceeded` which returns the latest tool output rather than 502'ing.

`sql_query` reads `runtime.context` via `ToolRuntime[ContextSchema]`, so the parent's `db_dsn` / dialect / tenant flow in automatically. `sql_interact.py` caches `SQLDatabase` / `SQLDatabaseToolkit` / tool tuples by DSN (`lru_cache(maxsize=16)`) and augments LangChain's stock toolkit with two extras: `get_schema` (a thin wrapper that calls `database.get_table_info`) and `find_slow_sql` (PostgreSQL-only, queries `pg_stat_statements`; returns a friendly error on other dialects). Schema/list/query observations are truncated by `_trim_observation` in `sql_node.py` (table list 4k, schema 12k, query result 8k chars) to keep prompts bounded.

### MCP (`askanswer/mcp.py`)

`MCPClientManager` runs a dedicated asyncio loop on a background thread. All `add_url`/`add_stdio`/`call_tool` calls hop onto that loop via `asyncio.run_coroutine_threadsafe` so each server's async context manager enters and exits in the same task — necessary to avoid anyio cancel-scope errors. Always go through `get_manager()` / `shutdown_manager()`; the latter is registered in `cli.main` with `atexit`. Tool specs returned by `list_tools()` are wrapped into `StructuredTool`s by `registry._wrap_mcp_tool` (which converts the JSON Schema to a flat pydantic model — non-trivial schemas with `$ref`/`anyOf` are skipped).

### Persistence (`askanswer/persistence.py`)

A single SQLite file (default `~/.askanswer/state.db`, override with `ASKANSWER_DB_PATH` or `XDG_DATA_HOME`) carries both LangGraph's `SqliteSaver` checkpoint tables and our `thread_meta` table (title, preview, last_intent, model_label, created_at, updated_at, message_count, tags). Both share one `sqlite3.Connection` opened with `check_same_thread=False`, WAL journal mode, `busy_timeout=5000`, and `synchronous=NORMAL` so two concurrent `askanswer` processes don't immediately hit `database is locked`. Schema versioning lives in our own `askanswer_schema` row (current version: `1`); add a `if current < N` branch in `_migrate` when adding columns. `get_persistence()` is a lazy singleton — never construct `PersistenceManager` directly, and never import this module from `graph.py` at top level (the `--graph` path explicitly avoids creating the DB by passing `InMemorySaver`).

### Human-in-the-loop shell

The `gen_shell_commands_run` tool is special-cased via `confirmation_class="shell"`. `_route_from_answer` sends the call to `_shell_plan_node`, which pre-generates the command and writes it to `state["pending_shell"]` (persisted by the parent checkpointer so replays after `interrupt()` don't re-call the LLM). `_tools_node` then calls `_run_with_confirmation`, which raises `interrupt()`. The CLI catches the interrupt in `stream_query`, prompts the user, and resumes via `Command(resume={"approve": ..., "command": ...})`. Dangerous commands are blocked by `_DANGEROUS_PATTERNS` in `tools.py` both before user prompt and after any user edit.

### CLI (`askanswer/cli.py`)

Hand-rolled REPL: ANSI styling in class `C`, double-width-aware padding (`_visual_width`), boxed input prompt, per-node progress markers (`⏺ Node(detail)`) rendered from `app.stream(stream_mode="updates")`. The intent label shown on the `understand` marker comes from `handler.cli_label(update)` via `_intent_cli_label` — don't hard-code intent strings here. Slash commands (`/help`, `/clear`, `/status`, `/model`, `/mcp …`, `/threads [keyword]`, `/resume <序号|id>`, `/title <name>`, `/delete <序号|id>`, `/exit`) are routed by `handle_command`; `_LAST_LIST` caches the most recent `/threads` output so `/resume 1` and `/delete 1` can address rows by index. The `!<cmd>` prefix bypasses the graph entirely and runs a shell command after a danger check.

### Model loading & swap (`askanswer/load.py`)

`model` is a `_ModelProxy` wrapping the real `init_chat_model(...)` backend. Modules import it once at startup; `set_model("provider:name")` (called by `/model`) builds a new backend and `_swap`s it in place via `object.__setattr__`, so every existing reference is automatically redirected. **Always import `model` from `.load` and call through it** — never cache the inner backend, or you'll keep talking to the old model after `/model`. `current_model_label()` returns the active `provider:spec` for diagnostics.

## Conventions

- Nodes return partial state dicts that the reducers merge — never mutate `state` in place. When sorcery decides to retry, write the structured `retry_directive` (dict); when answer consumes it, return `retry_directive: {}` to clear.
- Module-level side effects are forbidden in `schema.py` (no stub graphs); keep it pure data. `persistence.py` must stay lazy — don't import it from `graph.py` at module top.
- New tools should be registered through `registry.py` with the right tag set, not bound directly to the model. Anything needing user approval must set `confirmation_class="shell"` (or a future class) — not the legacy `requires_confirmation=True` (which is now a read-only compatibility property).
- New file extensions for `read_file` should be added to `FILE_EXTENSIONS` in `intents/base.py` so the file-path regex used by `FileReadHandler.local_classify` (and `IntentRegistry._fallback_classification`) picks them up.
- New intents go in `intents/<name>.py` + a line in `intents/__init__.py`. Lower `priority` runs first; place the global fallback heuristic in `IntentRegistry._fallback_classification`, not in any handler.
- Prefer the typed `ContextSchema` for per-invocation config; only fall back to env vars at the CLI boundary. Inside tools, read it via `ToolRuntime[ContextSchema]` and `normalize_context`.

## What's documented elsewhere

- `README.md` — user-facing install/usage, MCP examples, full tool table
- `CHANGELOG.md` — dated changelog (Added/Changed/Verified)
- `TODO.md` — prioritized backlog (P0 structured output / Literal types / LangSmith, P1 streaming / HITL unification, P2 testing) — read before suggesting larger refactors
- `docs/orchestration-extensibility-plan.md` — design rationale for the intent-handler refactor (and a self-review section)
- `docs/enterprise-persistence-plan.md` — phased plan for persistence (phase A shipped, phase B `/undo` pending)
- `docs/monitoring-plan.md`, `docs/enterprise-application-challenges.md`, `docs/phase-1-2-plan.md` — in-flight design notes; check these before proposing observability, audit, or subgraph-extraction work to avoid duplicating ideas already scoped
