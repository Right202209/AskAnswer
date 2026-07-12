# Architecture — Orchestration (request flow)

## Main graph (`askanswer/graph.py`)

Single-entry `create_search_assistant(checkpointer=None)` returns a compiled `StateGraph(SearchState, context_schema=ContextSchema)`. When `checkpointer is None` it lazy-imports `persistence.get_persistence().checkpointer` (a shared `SqliteSaver`); pass `InMemorySaver()` to bypass disk (used by `draw_search_assistant_mermaid` so `--graph` doesn't create an empty `state.db`). Topology is **intent-agnostic at the parent level**:

```
START → understand → answer → sorcery → (END | answer for retry)
```

All six intents (`file_read | sql | chat | search | math | helix`, plus any plugin-registered ones) flow through the same `answer` node (the compiled react subgraph). Intent only affects (a) the system-prompt hint produced by the active `IntentHandler.prompt_hint(state)`, and (b) which tool **tag set** the registry exposes (see `architecture-extensibility.md`). `tavily_search` and `read_file` are registered tools, not parent-graph nodes — the LLM decides when to call them. `route_from_sorcery` is the only conditional edge: when sorcery sets `step="retry_search"` it re-enters `answer`; the answer node consumes the structured `retry_directive` field (set by sorcery) and bakes it into the system prompt — there is no longer a synthetic `HumanMessage` injection.

## State (`askanswer/state.py`)

`SearchState` is a `TypedDict` with `messages` (LangGraph `add_messages` reducer), plus flat fields: `user_query`, `search_query`, `search_results`, `final_answer`, `intent`, `file_path`, `retry_count`, `step`, `retry_directive` (sorcery → answer hand-off, cleared after consumption), `pending_shell` and `pending_fs_write` (HITL persistence — one bucket per confirmation class). **When adding a new node that needs to persist data, add the field here** — nodes return partial dicts that get merged.

## Runtime context (`askanswer/schema.py`)

`ContextSchema` (dataclass: `llm_provider`, `db_dsn`, `db_dialect`, `tenant_id`) is the LangGraph **runtime context**, distinct from `SearchState`. The CLI builds it from env vars in `cli.py:_runtime_context()` and passes it via `app.stream(..., context=...)`. SQL nodes accept it as their second arg via `Runtime[ContextSchema]` and read it with `normalize_context(getattr(runtime, "context", None))` (which also accepts plain dicts). Use this for per-invocation config that shouldn't pollute the message-merging state.

## Parent-graph nodes (`askanswer/nodes.py`)

Two node bodies live here: `understand_query_node` (the first node) and `sorcery_answer_node` (the third). `understand_query_node` runs `_local_intent` → `_intent_from_llm` (structured output via `model.with_structured_output(IntentClassification)`) → `_local_intent(fallback=True)`, then writes `user_query`/`search_query`/`file_path`/`intent`/`retry_count=0`/`retry_directive={}`/`step="understood"` plus a one-line AIMessage hint. `sorcery_answer_node` looks up the active handler, returns `_finalize(state)` once `retry_count >= handler.max_retries`, otherwise calls `handler.evaluate(state)` and either finalizes (`decision == "pass"`) or writes `step="retry_search"` + `retry_directive` for the next answer pass. Evaluation logic itself does **not** live here — it's in each `IntentHandler.evaluate`.

## React subgraph (`askanswer/react.py`, `askanswer/_react_internals.py`)

`build_react_subgraph()` compiles the answer ⇄ tools / confirm_plan loop and is wired into the parent graph as the `answer` node. It shares `SearchState` with the parent and is compiled **without its own checkpointer** so `interrupt()` from the HITL flow propagates to the parent stream. `_react_internals.py` holds the actual node bodies (`_answer_node`, `_confirm_plan_node`, `_tools_node`, `_route_from_answer`, `_reclassify_intent`); treat it as private — call through `react.build_react_subgraph()`.

The subgraph entry is `START → clarify → answer` (`clarify` lives in `askanswer/clarify.py`, kept out of `_react_internals.py`). `clarify_node` runs the **generic clarification protocol**: only on the first answer pass (guarded by `step == "understood"`, so sorcery retries and the internal answer⇄tools loop skip it), it reads the already-committed `intent` (no LLM recompute) plus the runtime `ContextSchema`, asks the active handler for a `ClarificationRequest` (see architecture-extensibility), and if one comes back `interrupt({"type":"clarify", ...})`s — the CLI resolves it via `_prompt_clarification` (TTY menu / non-TTY default) and `Command(resume={"index","text"})`. The node merges the chosen `updates` into `SearchState` (e.g. fills `file_path`, flips `intent→chat`, narrows `user_query`) before `answer`. Parent topology is unchanged; when no handler needs clarification it returns `{}` at zero cost.

`_answer_node` (a) runs `_reclassify_intent` and writes the new intent back to `state["intent"]` if it changed, (b) calls `handler.prompt_hint(state)` for the system-prompt prefix, (c) appends `retry_directive["instruction"]` if present and clears the field on its next return, and (d) binds only `registry.list(tags=handler.bundle_tags)`.

`_tools_node` splits tool calls four ways: `plain_calls` go to `langgraph.prebuilt.ToolNode` (which auto-injects `ToolRuntime[ContextSchema]`); `shell_calls` (confirmation_class == "shell") go through `_run_shell_with_confirmation`; `fs_write_calls` (confirmation_class == "fs_write") go through `_run_fs_write_with_confirmation`; any other `confirmation_class` value lands in `unsupported_confirm_calls` and gets a `ToolMessage` explaining the executor isn't wired — the LLM can react instead of crashing. The "known executors" set lives in `_KNOWN_CONFIRM_CLASSES`; adding a new HITL class is a two-line addition there plus a new runner.

## Human-in-the-loop confirmations

Two classes are wired today: `shell` (`gen_shell_commands_run`) and `fs_write` (`write_file_patch`). `_route_from_answer` sends any tool call whose class is in `_KNOWN_CONFIRM_CLASSES` to `_confirm_plan_node`, which dispatches per call: shell goes through `_plan_shell_call` → `gen_shell_command_spec` → `pending_shell`; fs_write goes through `_plan_fs_write_call` → `propose_file_write` (path validation + diff vs current file) → `pending_fs_write`. The plans are persisted by the parent checkpointer so replays after `interrupt()` don't re-call the LLM / re-read disk. `_tools_node` then calls the matching `_run_*_with_confirmation`, which raises `interrupt({"type": "confirm_shell" | "confirm_fs_write", ...})`. The CLI catches the interrupt in `stream_query`, dispatches via `_prompt_confirmation` to the per-class UI, and resumes with `Command(resume={"approve": bool, ...})`. Danger filters (`_DANGEROUS_PATTERNS` for shell, `check_fs_write_dangerous` for fs_write) run both before the prompt and after any user edit, and again right before execution.
