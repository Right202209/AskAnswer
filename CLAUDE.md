# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Index only.** Read the linked topic for the task at hand. Paths are relative to the repository root, which contains `pyproject.toml`; `askanswer/` is the Python package directory.

## Sections

- [Commands and tests](.claude/mem/commands.md) — environment setup, packaging, linting, full-suite and single-test commands, test isolation.
- [Architecture and extension points](.claude/mem/architecture.md) — runtime entry points, graph/state flow, intent and tool registries, nested agents, interrupts, persistence, MCP.
- [Configuration and model calls](.claude/mem/configuration.md) — settings precedence, lazy model proxy, audit/telemetry, context-budget invariants.

## Maintaining this guidance

Edit the relevant `.claude/mem/` topic rather than duplicating its content here. Keep this index and `AGENTS.md` synchronized when changing topic paths or summaries. Base guidance on the current working tree, not removed CLI/server modules or missing historical documentation.
