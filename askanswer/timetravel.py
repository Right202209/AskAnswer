"""Checkpoint inspection, rewind, and fork helpers."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from itertools import islice
from typing import Any

from langchain_core.messages import HumanMessage

from .load import current_model_label
from .persistence import PersistenceManager


@dataclass
class CheckpointInfo:
    index: int
    node: str
    step: str
    message_count: int
    created_at: int
    pending_shell: bool
    snapshot: Any


def list_checkpoints(app, thread_id: str, *, limit: int = 50) -> list[CheckpointInfo]:
    config = {"configurable": {"thread_id": thread_id}}
    snapshots = list(islice(app.get_state_history(config), limit))
    out: list[CheckpointInfo] = []
    for index, snapshot in enumerate(snapshots):
        values = getattr(snapshot, "values", {}) or {}
        metadata = getattr(snapshot, "metadata", {}) or {}
        messages = values.get("messages") or []
        out.append(
            CheckpointInfo(
                index=index,
                node=_snapshot_node(metadata),
                step=str(values.get("step") or "—"),
                message_count=len(messages),
                created_at=_snapshot_ts(snapshot),
                pending_shell=bool(values.get("pending_shell")),
                snapshot=snapshot,
            )
        )
    return out


def rewind_to(app, thread_id: str, index: int) -> CheckpointInfo:
    """Create a new checkpoint from a historical snapshot."""
    current = app.get_state({"configurable": {"thread_id": thread_id}})
    current_values = getattr(current, "values", {}) or {}
    if current_values.get("pending_shell") or _snapshot_has_interrupt(current):
        raise RuntimeError("当前会话有挂起的 shell 确认，不能执行 /undo 或 /jump")

    checkpoints = list_checkpoints(app, thread_id, limit=max(index + 1, 50))
    if index < 0 or index >= len(checkpoints):
        raise IndexError("checkpoint 序号超出范围")
    target = checkpoints[index]
    values = dict(getattr(target.snapshot, "values", {}) or {})
    values["pending_shell"] = {}
    config = {"configurable": {"thread_id": thread_id}}
    _update_state(app, config, values)
    return target


def fork_thread(
    app,
    src_thread_id: str,
    persistence: PersistenceManager,
    *,
    index: int = 0,
) -> str:
    """Copy one snapshot into a new thread and return the new thread id."""
    checkpoints = list_checkpoints(app, src_thread_id, limit=max(index + 1, 50))
    if index < 0 or index >= len(checkpoints):
        raise IndexError("checkpoint 序号超出范围")
    values = dict(getattr(checkpoints[index].snapshot, "values", {}) or {})
    values["pending_shell"] = {}

    new_thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": new_thread_id}}
    _update_state(app, config, values)

    old = persistence.get_meta(src_thread_id)
    title = old.title if old and old.title else old.preview if old else None
    fork_title = f"{title or 'fork'} [fork]"
    messages = values.get("messages") or []
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    preview = _message_preview(human_messages[-1]) if human_messages else None
    persistence.upsert_meta(
        new_thread_id,
        title=fork_title[:120],
        intent=values.get("intent"),
        model_label=current_model_label(),
        preview=preview,
        message_count=len(human_messages),
    )
    return new_thread_id


def _update_state(app, config: dict, values: dict) -> None:
    try:
        app.update_state(config, values, as_node="sorcery")
    except TypeError:
        app.update_state(config, values)


def _snapshot_node(metadata: dict) -> str:
    writes = metadata.get("writes") if isinstance(metadata, dict) else None
    if isinstance(writes, dict) and writes:
        return ",".join(str(k) for k in writes.keys())
    source = metadata.get("source") if isinstance(metadata, dict) else None
    return str(source or "checkpoint")


def _snapshot_ts(snapshot) -> int:
    created = getattr(snapshot, "created_at", None)
    if isinstance(created, (int, float)):
        return int(created)
    if isinstance(created, str):
        try:
            # LangGraph commonly uses ISO 8601 strings.
            from datetime import datetime

            return int(datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp())
        except Exception:
            return 0
    return int(time.time())


def _snapshot_has_interrupt(snapshot) -> bool:
    for task in getattr(snapshot, "tasks", None) or ():
        if getattr(task, "interrupts", None):
            return True
    return False


def _message_preview(message: HumanMessage) -> str | None:
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return None
    return content.strip().replace("\n", " ")[:80] or None
