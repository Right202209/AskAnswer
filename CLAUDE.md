# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Index only.** This file is a table of contents. Open the linked file under `.claude/mem/` for the section you need — don't expect full context here.

## Sections

- [Commands & Required env](.claude/mem/commands.md) — install, run, env vars (`OPENAI_API_KEY`, `TAVILY_API_KEY`, `ASKANSWER_DB_PATH`, …)
- [Architecture — Orchestration](.claude/mem/architecture-orchestration.md) — main graph topology, `SearchState`, `ContextSchema`, parent nodes, react subgraph, HITL shell
- [Architecture — Extensibility](.claude/mem/architecture-extensibility.md) — `IntentHandler` protocol + `IntentRegistry`, `ToolRegistry` tags & confirmation classes
- [Architecture — Subgraphs as tools](.claude/mem/architecture-subgraphs.md) — SQL agent and Helix spec-loop, both exposed via the tool registry
- [Architecture — Runtime services](.claude/mem/architecture-runtime.md) — MCP client manager, SQLite persistence, CLI/REPL, `_ModelProxy` swap
- [Conventions](.claude/mem/conventions.md) — partial-dict returns, lazy persistence, registry-only tools, intent extension rules
- [What's documented elsewhere](.claude/mem/external-docs.md) — `README.md`, `CHANGELOG.md`, `TODO.md`, `docs/*-plan.md`

## Maintenance rule

When changing project docs:
1. Edit the relevant `.claude/mem/<topic>.md` — that's where the content lives.
2. Update the matching line above only if the topic, file path, or one-line hook changed.
3. `/init` (re)generates this file using the same index-plus-`.claude/mem/` structure; never collapse the content back into a single file.
