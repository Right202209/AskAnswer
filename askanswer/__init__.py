"""AskAnswer: an embeddable LangGraph agent core (可嵌入的 LangGraph agent 核心).

Public API (stable)::

    from askanswer import (
        ContextSchema, normalize_context,          # per-invocation runtime context
        create_search_assistant,                   # compile the agent graph
        run_leg, stream_leg, RunEvent,             # run one "leg" and consume events
        query_input, resume_input, thread_config,  # graph inputs / config helpers
    )

Importing this package has no side effects: it reads no configuration files,
does not modify ``os.environ``, opens no database and needs no API key.
Configuration (``.env`` + layered ``settings.json``) is loaded lazily on first
use; see :mod:`askanswer.config.settings`.

Package layout: :mod:`~askanswer.core` (graph, runner, ReAct subgraph),
:mod:`~askanswer.config`, :mod:`~askanswer.llm`, :mod:`~askanswer.intents`,
:mod:`~askanswer.tools`, :mod:`~askanswer.subagents`, :mod:`~askanswer.storage`,
:mod:`~askanswer.observability`.
"""

from .config.schema import ContextSchema, normalize_context
from .core.graph import create_search_assistant
from .core.runner import RunEvent, query_input, resume_input, run_leg, stream_leg, thread_config

__all__ = [
    "ContextSchema",
    "RunEvent",
    "create_search_assistant",
    "normalize_context",
    "query_input",
    "resume_input",
    "run_leg",
    "stream_leg",
    "thread_config",
]
