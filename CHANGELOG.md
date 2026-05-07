# Changelog

## 2026-05-06

### Added
- Added an intent handler registry under `askanswer/intents/` with handlers for chat, search, SQL, file reads, and a `math` smoke-test intent.
- Added structured `retry_directive` state for sorcery-driven retries without synthetic user messages.
- Added tag-based tool filtering and `confirmation_class` metadata while preserving `bundle=` compatibility.

### Changed
- Refactored intent classification and answer evaluation so new intents can be registered through handlers instead of editing core graph branches.
- Moved intent-specific ReAct prompt hints and CLI labels into handlers.
- Kept the main graph shape stable: `START -> understand -> answer -> sorcery -> (END | answer)`.

### Verified
- `python -m compileall askanswer`
- `python -m askanswer --graph -`
- Local smoke checks for `math`, `search`, `sql`, `chat`, and `file_read` classification.
