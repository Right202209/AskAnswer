# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Index only.** This file is a table of contents. Open the linked file under `.claude/mem/` for the section you need — don't expect full context here.

## Sections

- [Commands & Required env](.claude/mem/commands.md) — **activate `.venv/` before any `python`/`pip`/`askanswer`**; install/run, `--graph` export, env vars (`OPENAI_API_KEY`, `TAVILY_API_KEY`, `WLANGGRAPH_POSTGRES_DSN`, `ASKANSWER_DB_PATH`, …), full slash-command list
- [Architecture — Orchestration](.claude/mem/architecture-orchestration.md) — main graph topology, `SearchState`, `ContextSchema`, parent nodes, react subgraph, confirmation HITL flow (shell / fs_write / external_api_paid)
- [Architecture — Extensibility](.claude/mem/architecture-extensibility.md) — `IntentHandler` protocol + `IntentRegistry` priorities, `ToolRegistry` tags & confirmation classes, MCP tool wrapping
- [Architecture — Subgraphs as tools](.claude/mem/architecture-subgraphs.md) — SQL agent and Helix spec-loop, both exposed via the tool registry with `ToolRuntime` context passthrough
- [Architecture — Runtime services](.claude/mem/architecture-runtime.md) — `_ModelProxy` hot swap, audit/pricing, SQLite persistence, time travel, MCP client manager, CLI/REPL streaming
- [Conventions](.claude/mem/conventions.md) — partial-dict returns, registry-only tools, lazy persistence, intent extension rules, security boundaries
- [What's documented elsewhere](.claude/mem/external-docs.md) — `README.md`, `CHANGELOG.md`, `TODO.md`, `AGENTS.md`, `docs/*.md`, bug-risk + CLI-revamp reviews, `web/`

## Maintenance rule

When changing project docs:
1. Edit the relevant `.claude/mem/<topic>.md` — that's where the content lives.
2. Update the matching line above only if the topic, file path, or one-line hook changed.
3. `/init` (re)generates this file using the same index-plus-`.claude/mem/` structure; never collapse the content back into a single file.
