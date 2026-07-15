# Commands & Required env

## Python environment

**Always activate the project venv before running `python` / `pip` / `askanswer`.** The venv lives at `.venv/` at the repo root. From the repo root:

```bash
source .venv/bin/activate    # then python …, pip …, askanswer …
```

The `askanswer` entry point installed by `pip install -e .` only resolves inside the activated venv. Running the system `python` will miss `langchain` / `langgraph` / `prompt_toolkit` and look broken.

## Commands

- Install (editable, inside the activated venv): `pip install -e .` (Python 3.10+); add dev tooling with `pip install -e ".[dev]"` (pytest + ruff)
- Run CLI: `askanswer "<question>"` (single-shot) · `askanswer` (REPL) · `python -m askanswer`
- Run HTTP/SSE server: `python -m askanswer.server [--host 127.0.0.1] [--port 8765]` (or `askanswer-server` after reinstall; optional `ASKANSWER_SERVER_TOKEN` enables bearer auth — endpoints/events in `docs/important-documentation-c3-http-sse-server.md`)
- Export the LangGraph topology: `askanswer --graph` (stdout) or `askanswer --graph docs/graph.mmd`
- Test / lint (both run with **no API key** — LLM/network mocked, env isolated): `pytest -q` (107 cases across persistence/confirmations/mcp_profile/telemetry/registry/intents/graph) · `ruff check askanswer tests` (E/F/I baseline; `--fix` to autofix). Legacy spot check: `python -m compileall askanswer`.

## Required env (`.env` and/or `settings.json`)

`OPENAI_API_KEY`, `TAVILY_API_KEY` are required. Optional: `OPENWEATHER_API_KEY` (weather tool), `WLANGGRAPH_POSTGRES_DSN` + `ASKANSWER_DB_DIALECT` (SQL agent default DB), `ASKANSWER_TENANT_ID` (multi-tenant isolation — when set, `/threads` `/audit` `/usage` `/resume` `/delete` only see that tenant's rows; unset = all rows, non-regression), `ASKANSWER_DB_PATH` / `XDG_DATA_HOME` (override the SQLite state file location, defaults to `~/.askanswer/state.db`), `ASKANSWER_MCP_PROFILE` (override the MCP autoconnect profile path, defaults to `~/.askanswer/mcp.json`), `ASKANSWER_MCP_ALL_INTENTS=1` (expose MCP tools to every intent), `ASKANSWER_DEFAULT_MODEL` (startup model label, default `openai:gpt-5.4`), `ASKANSWER_SETTINGS` (override path of the **user** settings.json). Observability (all opt-in, zero-overhead when unset): `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY` (LangSmith exporter), `ASKANSWER_OTEL_EXPORTER` (OpenTelemetry exporter).

### settings.json (Claude Code–style layers)

Implemented in `askanswer/settings.py`, applied from `load.py` **before** `load_dotenv(override=True)`:

| Scope | Path | Notes |
| --- | --- | --- |
| User | `~/.askanswer/settings.json` (or `$XDG_CONFIG_HOME/askanswer/settings.json`; `ASKANSWER_SETTINGS` overrides) | personal defaults |
| Project | `<repo>/.askanswer/settings.json` | shareable; walk-up from cwd via `.git` / `pyproject.toml` / `.askanswer` |
| Local | `<repo>/.askanswer/settings.local.json` | personal; gitignored |

Global precedence (high → low): **process env → settings local → project → user → `.env` (lowest baseline)**. Implemented by `bootstrap_environ()` in `settings.py` (called from `load.py`): snapshot pre-existing keys → `load_dotenv(override=False)` → `apply_settings(override=True, protect=snapshot)`. Within settings files, merge order is user → project → local. First-class keys (`model`, `models`, `tenant_id`, `db_path`, `context`, `run_token_budget`, `mcp_all_intents`, …) expand to the same `ASKANSWER_*` env names; an explicit `env` block overrides those. See `settings.example.json`.

## Standalone MCP server

`python -m askanswer.helix_mcp` starts a stdio MCP server exposing the Helix spec-loop as `helix_spec_loop` for external agents (does NOT import `graph.py`/trigger persistence). Another AskAnswer instance can attach it with `/mcp add_stdio helix python -m askanswer.helix_mcp`.

## Slash commands (REPL)

`SLASH_COMMANDS` in `ui_input.py` is the single source of truth — keep that list and `help_block` in sync when adding one. Currently:

- Session: `/help [cmd]` · `/clear` · `/status` · `/model [provider:name]` · `/exit` (`/quit` `/q`)
- Threads: `/threads [keyword]` · `/resume <序号|id>` · `/title <name>` · `/delete <序号|id>` (all tenant-filtered)
- Time-travel: `/checkpoints` (shows label column) · `/undo [n] [--label NAME]` · `/jump <index>` · `/fork [index]` (see `timetravel.py`; `--label` names a restore point in `checkpoint_label`, `/undo --label NAME` resolves back to it)
- Audit / cost: `/audit [thread] [--kind …] [--limit N]` · `/usage [--days N] [--thread …]` (see `audit.py`, `pricing.py`)
- MCP: `/mcp <url> [name]` · `/mcp add_stdio <name> <cmd> [args…]` · `/mcp list [-v]` · `/mcp health [name]` · `/mcp tools [server]` · `/mcp remove <name>` (adds/removes persist to the autoconnect profile)
- Shell escape: `!<cmd>` runs a shell command directly, with danger-pattern checks.
