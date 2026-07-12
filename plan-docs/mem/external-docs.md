# What's documented elsewhere

- `README.md` — user-facing install/usage, MCP examples, full tool table
- `CHANGELOG.md` — dated changelog (Added/Changed/Verified)
- `TODO.md` — prioritized backlog (P0 structured output / Literal types / LangSmith, P1 streaming / HITL unification, P2 testing) — read before suggesting larger refactors
- `docs/orchestration-extensibility-plan.md` — design rationale for the intent-handler refactor (and a self-review section)
- `docs/helix-subgraph-plan.md` — design + implementation log for the Helix spec-evolution subgraph
- `docs/enterprise-persistence-plan.md` — phased plan for persistence (phase A shipped, phase B time-travel shipped — `/checkpoints` `/undo` `/jump` `/fork`)
- `docs/monitoring-plan.md`, `docs/enterprise-application-challenges.md`, `docs/phase-1-2-plan.md` — in-flight design notes; check these before proposing observability, audit, or subgraph-extraction work to avoid duplicating ideas already scoped
- `docs/bug-risk-review-2026-05-08.md` — module-level bug-risk audit; tightening of `read_file` / shell / sql / mcp / history boundaries was tracked here (commit `4030311`)
- `docs/review-1c210dc-cli-revamp.md` — review of the prompt_toolkit / spinner / live-token CLI revamp
- `docs/possibility-space-exploration.md` + `docs/possibility-space-execution-plan.md` — exploratory design notes (unscoped); skim before proposing new product directions
- `docs/claude-code-architecture-study.md`, `docs/cli-references-pi-mono-deepseek-tui.md` — external reference notes used while designing the CLI revamp
- `plan-docs/00-direction-and-goals.md` / `01-spec.md` / `02-execution-plan.md` — the authoritative replan (Land → Harden → Evolve): invariants, quality gates, per-step acceptance; progress boxes live in `02-execution-plan.md`
- `docs/important-documentation-c2-clarification.md` — C2 generic-clarification change log + design invariants (code landed, not yet run; checklist migrated to the verification matrix)
- `docs/important-documentation-c3-http-sse-server.md` — C3 HTTP/SSE server: API/event contract, security-checklist results (code landed, not yet run; checklist migrated to the verification matrix)
- `docs/important-documentation-verification-matrix.md` — **the master mandatory-verification matrix** for all uncommitted/unverified work: full change→commit mapping (incl. cli.py C2/C3 hunk split), ordered test groups G0–G7 with item IDs, commit gates + failure protocol; check boxes ONLY here
