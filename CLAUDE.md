# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Install (editable): `pip install -e .` (Python 3.10+)
- Run CLI: `askanswer "<question>"` (single-shot) · `askanswer` (REPL) · `python -m askanswer`
- Export the LangGraph topology: `askanswer --graph` (stdout) or `askanswer --graph docs/graph.mmd`
- No test suite or linter is wired up yet; do not invent commands for them.

## Required env (`.env` at repo root)

`OPENAI_API_KEY`, `TAVILY_API_KEY` are required. Optional: `OPENWEATHER_API_KEY` (weather tool), `WLANGGRAPH_POSTGRES_DSN` + `ASKANSWER_DB_DIALECT` (SQL agent default DB), `ASKANSWER_TENANT_ID`.

## Architecture

### Main graph (`askanswer/graph.py`)

Single-entry `create_search_assistant()` returns a compiled `StateGraph(SearchState, context_schema=ContextSchema)` with an in-memory checkpointer. Topology:

```
START → understand ─┬─ file_read ─────────────────→ sorcery ─→ END
                    ├─ answer (chat | sql) ───────→ sorcery
                    └─ search → answer ───────────→ sorcery (may re-search once)
```

`route_from_understand` reads `state["intent"]` (`file_read | sql | chat | search`); `sql` and `chat` both flow into the `answer` node — what differs is which tool **bundle** the registry exposes (see Tool registry below). The `answer` node is the compiled subgraph from `react.py` (answer ⇄ tools / shell_plan).

### State (`askanswer/state.py`)

`SearchState` is a `TypedDict` with `messages` (LangGraph `add_messages` reducer), plus flat fields: `user_query`, `search_query`, `search_results`, `final_answer`, `intent`, `file_path`, `retry_count`, `step`, `pending_shell`. **When adding a new node that needs to persist data, add the field here** — nodes return partial dicts that get merged.

### Runtime context (`askanswer/schema.py`)

`ContextSchema` (dataclass: `llm_provider`, `db_dsn`, `db_dialect`, `tenant_id`) is the LangGraph **runtime context**, distinct from `SearchState`. The CLI builds it from env vars in `cli.py:_runtime_context()` and passes it via `app.stream(..., context=...)`. SQL nodes accept it as their second arg via `Runtime[ContextSchema]` and read it with `normalize_context(getattr(runtime, "context", None))` (which also accepts plain dicts). Use this for per-invocation config that shouldn't pollute the message-merging state.

### Intent classification (`askanswer/nodes.py`)

`understand_query_node` runs a **fast local classifier first** (`_local_intent`: regex + keyword tables for file paths, SQL, search, chat). Only ambiguous inputs hit the LLM (`_intent_from_llm`, JSON-output prompt). LLM failures fall back to `_local_intent(..., fallback=True)`. When extending intents, update `_INTENTS`, the relevant keyword tuple (`_SQL_KEYWORDS`, `_SEARCH_KEYWORDS`, …), and `route_from_understand` in `graph.py`.

### React subgraph (`askanswer/react.py`, `askanswer/_react_internals.py`)

`build_react_subgraph()` compiles the answer ⇄ tools / shell_plan loop and is wired into the parent graph as the `answer` node. It shares `SearchState` with the parent and is compiled **without its own checkpointer** so `interrupt()` from the shell HITL flow propagates to the parent stream. `_react_internals.py` holds the actual node bodies (`_answer_node`, `_shell_plan_node`, `_tools_node`, `_route_from_answer`); treat it as private — call through `react.build_react_subgraph()`.

`_tools_node` splits tool calls into plain (delegated to `langgraph.prebuilt.ToolNode`, which auto-injects `ToolRuntime[ContextSchema]`) and confirmation-required (run via `_run_with_confirmation`, which raises `interrupt()`).

### Tool registry (`askanswer/registry.py`)

Process-wide `ToolRegistry` (`get_registry()`) is the single source of truth for which tools the LLM can see. Each `ToolDescriptor` carries a `frozenset` of **bundles** (`chat | search | file_read | sql`) and the `_answer_node` filters by the active intent: `registry.list(bundle=intent)`. Seeded on first use with built-in tools (`_seed_builtin`), the SQL natural-language tool (`_seed_sql`), and any live MCP tools (`refresh_mcp`). Confirmation-required tools (currently only `gen_shell_commands_run`) set `requires_confirmation=True`; `confirmation_names()` is what `_route_from_answer` and `_tools_node` use to detect HITL routing. After connecting/disconnecting an MCP server, call `registry.refresh_mcp()` to rebuild the `mcp:*` slice.

### SQL agent (`askanswer/sqlagent/`)

The SQL flow is **exposed as a tool**, not as a parent-graph node. `sql_tool.sql_query` is the registry entry (bundle: `chat | sql`); when invoked, it calls `sql_agent.run_sql_agent` which executes a separate `StateGraph(SqlAgentState, context_schema=ContextSchema)` from `build_sql_agent()`. Internal flow: `list_tables → call_get_schema → get_schema → generate_query → (check_query → run_query → generate_query)*`. Hard caps: `MAX_SQL_QUERY_CALLS = 2`, `SQL_RECURSION_LIMIT = 12` — when exceeded, routes to `limit_exceeded` which returns the latest tool output rather than 502'ing.

`sql_query` reads `runtime.context` via `ToolRuntime[ContextSchema]`, so the parent's `db_dsn` / dialect / tenant flow in automatically. `sql_interact.py` caches `SQLDatabase` / `SQLDatabaseToolkit` / tool tuples by DSN (`lru_cache(maxsize=16)`). Schema/list/query observations are truncated by `_trim_observation` in `sql_node.py` (table list 4k, schema 12k, query result 8k chars) to keep prompts bounded.

### MCP (`askanswer/mcp.py`)

`MCPClientManager` runs a dedicated asyncio loop on a background thread. All `add_url`/`add_stdio`/`call_tool` calls hop onto that loop via `asyncio.run_coroutine_threadsafe` so each server's async context manager enters and exits in the same task — necessary to avoid anyio cancel-scope errors. Always go through `get_manager()` / `shutdown_manager()`; the latter is registered in `cli.main` with `atexit`. Tool specs returned by `list_tools()` are wrapped into `StructuredTool`s by `registry._wrap_mcp_tool` (which converts the JSON Schema to a flat pydantic model — non-trivial schemas with `$ref`/`anyOf` are skipped).

### Human-in-the-loop shell

The `gen_shell_commands_run` tool is special-cased via `requires_confirmation=True`. `_route_from_answer` sends the call to `_shell_plan_node`, which pre-generates the command and writes it to `state["pending_shell"]` (persisted by the parent checkpointer so replays after `interrupt()` don't re-call the LLM). `_tools_node` then calls `_run_with_confirmation`, which raises `interrupt()`. The CLI catches the interrupt in `stream_query`, prompts the user, and resumes via `Command(resume={"approve": ..., "command": ...})`. Dangerous commands are blocked by `_DANGEROUS_PATTERNS` in `tools.py` both before user prompt and after any user edit.

### CLI (`askanswer/cli.py`)

Hand-rolled REPL: ANSI styling in class `C`, double-width-aware padding (`_visual_width`), boxed input prompt, per-node progress markers (`⏺ Node(detail)`) rendered from `app.stream(stream_mode="updates")`. Slash commands (`/help`, `/clear`, `/status`, `/mcp …`, `/exit`) are routed by `handle_command`. The `!<cmd>` prefix bypasses the graph entirely and runs a shell command after a danger check.

## Conventions

- Nodes return partial state dicts that the reducers merge — never mutate `state` in place.
- Module-level side effects are forbidden in `schema.py` (no stub graphs); keep it pure data.
- New tools should be registered through `registry.py` with the right bundle set, not bound directly to the model. Anything needing user approval must set `requires_confirmation=True`.
- New file extensions for `read_file` should also be added to `_FILE_EXTENSIONS` in `nodes.py` so the local intent classifier can route them.
- Prefer the typed `ContextSchema` for per-invocation config; only fall back to env vars at the CLI boundary. Inside tools, read it via `ToolRuntime[ContextSchema]` and `normalize_context`.

## What's documented elsewhere

- `README.md` — user-facing install/usage, MCP examples, full tool table
- `TODO.md` — prioritized backlog (P0 persistence, P1 streaming/HITL/graph cleanup, P2 testing) — read before suggesting larger refactors
