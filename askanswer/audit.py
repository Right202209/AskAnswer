"""Lightweight audit logging for CLI runs.

The graph should not carry telemetry in ``SearchState``.  This module keeps the
current run context outside the graph and flushes events into ``audit_event`` at
the CLI boundary.
"""

from __future__ import annotations

import json
import os
import time
from contextvars import ContextVar, Token
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .persistence import get_persistence

_THREAD_ID: ContextVar[str | None] = ContextVar("askanswer_audit_thread_id", default=None)
_PENDING: ContextVar[list[dict] | None] = ContextVar("askanswer_audit_pending", default=None)
_TENANT_ID: ContextVar[str | None] = ContextVar("askanswer_audit_tenant_id", default=None)
_FALLBACK_THREAD_ID: str | None = None


def begin_run(thread_id: str, *, tenant_id: str | None = None) -> tuple[Token, Token, Token]:
    """Start collecting audit events for one graph run.

    ``tenant_id`` is captured for the whole run so every event is attributed to
    the same tenant even if the process-wide env var changes mid-run. When not
    given, it falls back to ``ASKANSWER_TENANT_ID`` (may be ``None``).
    """
    global _FALLBACK_THREAD_ID
    _FALLBACK_THREAD_ID = thread_id
    tenant = tenant_id if tenant_id is not None else (os.environ.get("ASKANSWER_TENANT_ID") or None)
    return (
        _THREAD_ID.set(thread_id),
        _PENDING.set([]),
        _TENANT_ID.set(tenant),
    )


def end_run(tokens: tuple[Token, Token, Token]) -> None:
    """Restore the previous audit context."""
    global _FALLBACK_THREAD_ID
    thread_token, pending_token, tenant_token = tokens
    _TENANT_ID.reset(tenant_token)
    _PENDING.reset(pending_token)
    _THREAD_ID.reset(thread_token)
    _FALLBACK_THREAD_ID = _THREAD_ID.get()


def current_thread_id() -> str | None:
    return _THREAD_ID.get() or _FALLBACK_THREAD_ID


def current_tenant_id() -> str | None:
    return _TENANT_ID.get()


def summarize_args(args: Any, *, limit: int = 200) -> str:
    """Serialize tool args into a compact, bounded string."""
    try:
        text = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(args)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run_usage_so_far() -> tuple[int, int]:
    """当前 run 已累计的 ``(input_tokens, output_tokens)``。

    只读本次 run 的内存缓冲（挂起事件），不查库 —— 供 sorcery 的成本闸门在
    「是否继续质量重试」前做零开销判断。缓冲不存在（begin_run 之外）时返回 0。
    """
    pending = _PENDING.get() or []
    input_total = 0
    output_total = 0
    for event in pending:
        if event.get("kind") != "llm_call":
            continue
        input_total += event.get("input_tokens") or 0
        output_total += event.get("output_tokens") or 0
    return input_total, output_total


def log_event(
    *,
    kind: str,
    thread_id: str | None = None,
    tool_name: str | None = None,
    args_summary: str | None = None,
    result_size: int | None = None,
    model_label: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cached_tokens: int | None = None,
    duration_ms: int | None = None,
    intent: str | None = None,
    error: str | None = None,
    immediate: bool = False,
) -> None:
    """Queue or immediately persist one audit event."""
    tid = thread_id or current_thread_id()
    if not tid:
        return
    event = {
        "thread_id": tid,
        "ts": int(time.time()),
        "kind": kind,
        "tool_name": tool_name,
        "args_summary": args_summary,
        "result_size": result_size,
        "model_label": model_label,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        # cached_tokens 目前仅在内存事件里流转（成本闸门/回归报告可读），
        # 落库需要 schema v5 加列 —— 见 important-documentation-d1 的后续项。
        "cached_tokens": cached_tokens,
        "duration_ms": duration_ms,
        "intent": intent,
        "error": error,
        "tenant_id": current_tenant_id(),
    }
    pending = _PENDING.get()
    if immediate or pending is None:
        _persist(event)
        return
    pending.append(event)


def flush_pending(*, thread_id: str | None = None, intent: str | None = None) -> int:
    """Flush queued events into SQLite and return the number written."""
    pending = _PENDING.get()
    if not pending:
        return 0
    for event in pending:
        if thread_id:
            event["thread_id"] = thread_id
        if intent and not event.get("intent"):
            event["intent"] = intent
    try:
        count = get_persistence().log_audit_events(pending)
    except Exception:
        # Audit must never break the user-facing answer path.
        count = 0
    pending.clear()
    return count


def _persist(event: dict) -> None:
    try:
        get_persistence().log_audit_event(
            event["thread_id"],
            ts=event.get("ts"),
            kind=event["kind"],
            tool_name=event.get("tool_name"),
            args_summary=event.get("args_summary"),
            result_size=event.get("result_size"),
            model_label=event.get("model_label"),
            input_tokens=event.get("input_tokens"),
            output_tokens=event.get("output_tokens"),
            duration_ms=event.get("duration_ms"),
            intent=event.get("intent"),
            error=event.get("error"),
            tenant_id=event.get("tenant_id"),
        )
    except Exception:
        # Audit must never break the user-facing answer path.
        return


class LLMUsageCallback(BaseCallbackHandler):
    """Collect token usage and latency for one LangChain model invocation."""

    def __init__(self, *, model_label: str):
        self.model_label = model_label
        self.started_at = time.monotonic()

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.started_at = time.monotonic()

    def on_llm_start(self, *args, **kwargs) -> None:
        self.started_at = time.monotonic()

    def on_llm_end(self, response, **kwargs) -> None:
        input_tokens, output_tokens, cached_tokens = _extract_usage(response)
        log_event(
            kind="llm_call",
            model_label=self.model_label,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            duration_ms=int((time.monotonic() - self.started_at) * 1000),
        )

    def on_llm_error(self, error, **kwargs) -> None:
        log_event(
            kind="llm_call",
            model_label=self.model_label,
            duration_ms=int((time.monotonic() - self.started_at) * 1000),
            error=str(error),
        )


def with_llm_audit_callback(config: Any, *, model_label: str) -> dict:
    """Return a RunnableConfig-like dict with our callback appended."""
    cfg = dict(config or {})
    callbacks = cfg.get("callbacks")
    if callbacks is None:
        merged = []
    elif isinstance(callbacks, (list, tuple)):
        merged = list(callbacks)
    else:
        merged = [callbacks]
    merged.append(LLMUsageCallback(model_label=model_label))
    cfg["callbacks"] = merged
    return cfg


def _extract_usage(response) -> tuple[int | None, int | None, int | None]:
    """Best-effort extraction across LangChain provider variants.

    返回 ``(input, output, cached_input)``。cached_input 兼容三种来源：
    OpenAI ``prompt_tokens_details.cached_tokens``、Anthropic
    ``cache_read_input_tokens``、LangChain 标准 ``input_token_details.cache_read``。

    Iterates candidates lazily so a provider that puts usage in ``llm_output``
    doesn't pay the cost of scanning every generation message.
    """
    for usage in _candidate_usages(response):
        input_tokens = _first_int(
            usage,
            "input_tokens",
            "prompt_tokens",
            "prompt_token_count",
            "input_token_count",
        )
        output_tokens = _first_int(
            usage,
            "output_tokens",
            "completion_tokens",
            "completion_token_count",
            "output_token_count",
        )
        if input_tokens is None and output_tokens is None:
            continue
        cached_tokens = _first_int(
            usage,
            "cache_read_input_tokens",
            ("input_token_details", "cache_read"),
            ("prompt_tokens_details", "cached_tokens"),
        )
        return input_tokens, output_tokens, cached_tokens
    return None, None, None


def _candidate_usages(response):
    """Yield usage dicts in priority order: top-level llm_output, then per-generation."""
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        for key in ("token_usage", "usage", "usage_metadata"):
            value = llm_output.get(key)
            if isinstance(value, dict):
                yield value

    for group in getattr(response, "generations", None) or ():
        for generation in group or ():
            message = getattr(generation, "message", None)
            if message is None:
                continue
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, dict):
                yield usage
            metadata = getattr(message, "response_metadata", None)
            if isinstance(metadata, dict):
                for key in ("token_usage", "usage", "usage_metadata"):
                    value = metadata.get(key)
                    if isinstance(value, dict):
                        yield value


def _first_int(mapping: dict, *keys) -> int | None:
    """按序取第一个能转成 int 的键；键可以是 str，也可以是 (外层, 内层) 路径元组。"""
    for key in keys:
        path = key if isinstance(key, tuple) else (key,)
        value: Any = mapping
        for part in path:
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
