# Architecture — Subgraphs as tools

## SQL agent (`askanswer/sqlagent/`)

The SQL flow is **exposed as a tool**, not as a parent-graph node. `sql_tool.sql_query` is the registry entry (tags: `chat | sql | sql_tool`); when invoked, it calls `sql_agent.run_sql_agent` which executes a separate `StateGraph(SqlAgentState, context_schema=ContextSchema)` from `build_sql_agent()`. Internal flow: `list_tables → call_get_schema → get_schema → generate_query → (check_query → run_query → generate_query)*`. Hard caps: `MAX_SQL_QUERY_CALLS = 2`, `SQL_RECURSION_LIMIT = 12` — when exceeded, routes to `limit_exceeded` which returns the latest tool output rather than 502'ing.

`sql_query` reads `runtime.context` via `ToolRuntime[ContextSchema]`, so the parent's `db_dsn` / dialect / tenant flow in automatically. `sql_interact.py` caches `SQLDatabase` / `SQLDatabaseToolkit` / tool tuples by a **`(tenant_id, dsn)` key** (`lru_cache(maxsize=16)`, built by `_cache_key`) — two tenants sharing a DSN still get independent engines/pools, no cross-tenant connection reuse. It augments LangChain's stock toolkit with two extras: `get_schema` (a thin wrapper that calls `database.get_table_info`) and `find_slow_sql` (PostgreSQL-only, queries `pg_stat_statements`; returns a friendly error on other dialects). Schema/list/query observations are truncated by `_trim_observation` in `sql_node.py` (table list 4k, schema 12k, query result 8k chars) to keep prompts bounded.

## Helix subgraph (`askanswer/helix/`)

Same "subgraph-as-tool" shape as the SQL agent. `helix_spec_loop` (registered in `registry.py:_seed_helix` with tags `chat | helix | helix_tool`) calls `helix.agent.run_helix_agent`, which executes a `StateGraph(HelixState, context_schema=ContextSchema)` from `build_helix_agent()`. Topology: `interview → seed → execute → evaluate → (seed | finalize)`. Hard caps: `MAX_GENERATIONS = 3` (in `nodes.py`), `RECURSION_LIMIT = 24` — when the cap is reached, `route_after_evaluate` ends the loop and `finalize_verdict` writes `verdict="exhausted"` for the caller.

Every node uses `model.with_structured_output(<Pydantic>)` (`InterviewOutput` / `SeedOutput` / `ExecuteOutput` / `EvaluateOutput`) to avoid regex parsing of LLM output. `seed_node` writes a snapshot to `lineage` each generation so the final summary can show `gen1 → gen2 → gen3`. The tool returns a Markdown string (Goal / Constraints / Acceptance criteria / Artifact / Evaluation / Lineage) built by `format_helix_summary`.

**Interview is interactive** (`interview_node` in `helix/nodes.py`): each `InterviewQA` carries `question`, `options` (2–5 LLM-suggested choices) and a `default_answer` (assumption-tagged). `_ask_user` calls `ui_select.select_option` to render an arrow-key menu in TTY mode with an extra "其他（手动输入）" entry that opens a free-text prompt. When stdin is not a TTY (pipe, CI, non-interactive run) the node falls back to the LLM-provided `default_answer` so the loop never blocks. `_ask_user` returns the chosen option / typed text / default, and the result is stored as `qa[i]["a"]`.

`HelixHandler` triggers in three ways (priority 22, between SQL=20 and math=25):
1. English keywords (`helix`, `socratic`, `spec-first`, …),
2. Chinese keywords (`苏格拉底`, `需求澄清`, …),
3. **Ambiguity heuristic** — short text containing vague verbs (`做一个 / 搞个 / build a …`) without specificity markers (SQL keywords, URLs, file paths, math operators). The heuristic lives in `intents/helix._looks_ambiguous`; the LLM intent classifier (`nodes._intent_from_llm`) also lists `helix` as a choice for ambiguous requirements so non-keyword phrasing still routes here.

The handler's `evaluate()` is a no-op pass — the subgraph already self-evaluates, so the parent sorcery node doesn't retry.

Helix is **also exposed as a standalone MCP server**: `askanswer/helix_mcp.py` wraps `run_helix_agent` in a `FastMCP("askanswer-helix")` tool and runs it over stdio (`python -m askanswer.helix_mcp`). It reuses `helix.agent` only — deliberately no `graph.py` import, so it never triggers persistence init; in the non-TTY server process `interview_node` falls back to `default_answer`.

## research_brief subgraph (`askanswer/research/`)

Same subgraph-as-tool shape. `research_brief_loop` (registered in `registry._seed_research`, tags `chat | research | research_tool`) calls `research.agent.run_research_agent`, a **linear** `StateGraph(ResearchState)`: `plan_queries → search → synthesize → source_check`. `plan_queries` structured-outputs up to `MAX_QUERIES = 5` distinct queries; `search_node` runs each **through the registry's `tavily_search` tool** (not the raw client — keeps the search backend swappable) and regex-extracts source URLs from the result text; `synthesize` writes a brief with inline `[n]` citations; `source_check` cross-verifies claims against sources and emits the final references list. Output Markdown: Research Brief / References / Queries. `ResearchHandler` (priority 28, `intents/research.py`) triggers on research/调研 keywords; `bundle_tags = {research, search, research_tool, tavily}`, `max_retries = 2`.

## decision_memo subgraph (`askanswer/decision/`)

`decision_memo_loop` (registered in `registry._seed_decision`, tags `chat | decision | decision_tool`) calls `decision.agent.run_decision_agent`, a linear `StateGraph(DecisionState)`: `interview → decide`. **`interview` reuses Helix's `interview_node`** for Socratic clarification (so non-TTY falls back to `default_answer`); `decide_node` structured-outputs a `MemoOutput` (goal / constraints / 2–4 options with pros+cons / recommendation + rationale). Output Markdown: Goal / Constraints / Options / Recommendation. `DecisionHandler` (priority 23, `intents/decision.py`) triggers on decision/选型/权衡 keywords; `max_retries = 0` (subgraph is self-contained, parent sorcery doesn't retry).
