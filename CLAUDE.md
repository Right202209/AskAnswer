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
START → understand ─┬─ file_read ─→ sorcery ─→ END
                    ├─ sql ─→ END
                    ├─ answer (chat) ─⇄ tools / shell_plan ─→ sorcery
                    └─ search → answer ─⇄ tools / shell_plan ─→ sorcery (may re-search once)
```

`route_from_understand` reads `state["intent"]` (`file_read | sql | chat | search`) — the shape of branching lives here, not in the nodes. `route_from_answer` distinguishes shell tool calls (which go through `shell_plan` → HITL) from regular tools.

### State (`askanswer/state.py`)

`SearchState` is a `TypedDict` with `messages` (LangGraph `add_messages` reducer), plus flat fields: `user_query`, `search_query`, `search_results`, `final_answer`, `intent`, `file_path`, `retry_count`, `step`, `pending_shell`. **When adding a new node that needs to persist data, add the field here** — nodes return partial dicts that get merged.

### Runtime context (`askanswer/schema.py`)

`ContextSchema` (dataclass: `llm_provider`, `db_dsn`, `db_dialect`, `tenant_id`) is the LangGraph **runtime context**, distinct from `SearchState`. The CLI builds it from env vars in `cli.py:_runtime_context()` and passes it via `app.stream(..., context=...)`. SQL nodes accept it as their second arg via `Runtime[ContextSchema]` and read it with `normalize_context(getattr(runtime, "context", None))` (which also accepts plain dicts). Use this for per-invocation config that shouldn't pollute the message-merging state.

### Intent classification (`askanswer/nodes.py`)

`understand_query_node` runs a **fast local classifier first** (`_local_intent`: regex + keyword tables for file paths, SQL, search, chat). Only ambiguous inputs hit the LLM (`_intent_from_llm`, JSON-output prompt). LLM failures fall back to `_local_intent(..., fallback=True)`. When extending intents, update `_INTENTS`, the relevant keyword tuple (`_SQL_KEYWORDS`, `_SEARCH_KEYWORDS`, …), and `route_from_understand` in `graph.py`.

### SQL subgraph (`askanswer/sqlagent/`)

Separate `StateGraph(SqlAgentState, context_schema=ContextSchema)` built in `sql_agent.build_sql_agent()`. Flow: `list_tables → call_get_schema → get_schema → generate_query → (check_query → run_query → generate_query)*`. Hard caps: `MAX_SQL_QUERY_CALLS = 2`, `SQL_RECURSION_LIMIT = 12` — when exceeded, routes to `limit_exceeded` which returns the latest tool output rather than 502'ing.

`sql_interact.py` caches `SQLDatabase` / `SQLDatabaseToolkit` / tool tuples by DSN (`lru_cache(maxsize=16)`), so each tenant's DB gets its own engine. Schema/list/query observations are truncated by `_trim_observation` in `sql_node.py` (table list 4k, schema 12k, query result 8k chars) to keep prompts bounded.

### Tools and MCP (`askanswer/tools.py`, `askanswer/mcp.py`)

Built-in tools (`tools` list) are bound to the model in `generate_answer_node`. MCP tools are merged in as OpenAI-format function specs via `_mcp_tool_specs()`. `tools_node` dispatches: built-in via `tools_by_name`, MCP via `_mcp_manager().call_tool(name, args)` where `name` is the qualified `<server>__<tool>`.

`MCPClientManager` runs a dedicated asyncio loop on a background thread (`mcp.py`). All `add_url`/`add_stdio`/`call_tool` calls hop onto that loop via `asyncio.run_coroutine_threadsafe` so each server's async context manager enters and exits in the same task — necessary to avoid anyio cancel-scope errors. Always go through `get_manager()` / `shutdown_manager()`; the latter is registered in `cli.main` with `atexit`.

### Human-in-the-loop shell (`nodes.py`, `cli.py`)

The `gen_shell_commands_run` tool is special-cased: `route_from_answer` routes shell tool calls to `shell_plan_node`, which pre-generates the command and writes it to `state["pending_shell"]` (persisted by the checkpointer so replays after `interrupt()` don't re-call the LLM). `tools_node` then calls `_run_shell_with_confirmation`, which raises `interrupt()`. The CLI catches the interrupt in `stream_query`, prompts the user, and resumes via `Command(resume={"approve": ..., "command": ...})`. Dangerous commands are blocked by `_DANGEROUS_PATTERNS` in `tools.py` both before and after edit.

### CLI (`askanswer/cli.py`)

Hand-rolled REPL: ANSI styling in class `C`, double-width-aware padding (`_visual_width`), boxed input prompt, per-node progress markers (`⏺ Node(detail)`) rendered from `app.stream(stream_mode="updates")`. Slash commands (`/help`, `/clear`, `/status`, `/mcp …`, `/exit`) are routed by `handle_command`. The `!<cmd>` prefix bypasses the graph entirely and runs a shell command after a danger check.

## Conventions

- Nodes return partial state dicts that the reducers merge — never mutate `state` in place.
- Module-level side effects are forbidden in `schema.py` (no stub graphs); keep it pure data.
- New file extensions for `read_file` should also be added to `_FILE_EXTENSIONS` in `nodes.py` so the local intent classifier can route them.
- Prefer the typed `ContextSchema` for per-invocation config; only fall back to env vars at the CLI boundary.

## What's documented elsewhere

- `README.md` — user-facing install/usage, MCP examples, full tool table
- `TODO.md` — prioritized backlog (P0 persistence, P1 streaming/HITL/graph cleanup, P2 testing) — read before suggesting larger refactors
