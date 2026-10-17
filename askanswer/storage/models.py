"""存储层的数据视图与行转换（无 I/O）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ThreadMeta:
    """``thread_meta`` 表的 Python 视图。"""

    thread_id: str
    title: str | None
    tags: list[str] = field(default_factory=list)
    created_at: int = 0          # epoch seconds
    updated_at: int = 0          # epoch seconds
    message_count: int = 0       # 用户回合数（HumanMessage 数）
    last_intent: str | None = None
    model_label: str | None = None
    preview: str | None = None   # 最近一条用户消息前 80 字符
    tenant_id: str | None = None  # 归属租户；NULL 视为“不分租户”桶


@dataclass
class AuditEvent:
    """``audit_event`` 表的一行。"""

    id: int
    thread_id: str
    ts: int
    kind: str
    tool_name: str | None = None
    args_summary: str | None = None
    result_size: int | None = None
    model_label: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int | None = None
    intent: str | None = None
    error: str | None = None
    tenant_id: str | None = None


def derive_title(preview: str | None) -> str | None:
    """从 preview 抽取一行作为默认 title（首次插入时使用）。"""
    if not preview:
        return None
    line = preview.strip().splitlines()[0] if preview.strip() else ""
    return line[:30] or None


def clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def int_or_none(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def row_to_meta(row) -> ThreadMeta:
    """把 SELECT 出来的 row 还原为 ``ThreadMeta``。"""
    (
        thread_id,
        title,
        tags_json,
        created_at,
        updated_at,
        message_count,
        last_intent,
        model_label,
        preview,
        tenant_id,
    ) = row
    try:
        tags = json.loads(tags_json) if tags_json else []
    except json.JSONDecodeError:
        tags = []
    if not isinstance(tags, list):
        tags = []
    return ThreadMeta(
        thread_id=thread_id,
        title=title,
        tags=tags,
        created_at=int(created_at or 0),
        updated_at=int(updated_at or 0),
        message_count=int(message_count or 0),
        last_intent=last_intent,
        model_label=model_label,
        preview=preview,
        tenant_id=tenant_id,
    )


def row_to_audit(row) -> AuditEvent:
    (
        event_id,
        thread_id,
        ts,
        kind,
        tool_name,
        args_summary,
        result_size,
        model_label,
        input_tokens,
        output_tokens,
        duration_ms,
        intent,
        error,
        tenant_id,
    ) = row
    return AuditEvent(
        id=int(event_id or 0),
        thread_id=thread_id,
        ts=int(ts or 0),
        kind=kind,
        tool_name=tool_name,
        args_summary=args_summary,
        result_size=int_or_none(result_size),
        model_label=model_label,
        input_tokens=int_or_none(input_tokens),
        output_tokens=int_or_none(output_tokens),
        duration_ms=int_or_none(duration_ms),
        intent=intent,
        error=error,
        tenant_id=tenant_id,
    )
