# 会话搬运命令：``/export``（导出 md/json）、``/import``（从 json 导入为新会话）。
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)

from ...load import current_model_label
from ...persistence import AuditEvent, ThreadMeta, get_persistence
from ...timetravel import _update_state
from ..render import render_error
from ..text import _format_ts, _truncate
from ..theme import C
from ._common import _current_tenant, _resolve_thread_or_current, _split_args


def handle_export_command(args: str, *, current: str, app=None) -> None:
    if app is None:
        render_error("/export 只能在已初始化的图上使用")
        return
    parsed = _parse_export_args(args, current=current)
    if parsed is None:
        return
    target, fmt, out_path = parsed
    try:
        state = app.get_state({"configurable": {"thread_id": target.thread_id}})
        values = getattr(state, "values", {}) or {}
        messages = list(values.get("messages") or [])
        events = get_persistence().list_audit_events(
            thread_id=target.thread_id,
            limit=1000,
        )
    except Exception as exc:
        render_error(f"导出失败: {exc}")
        return

    if fmt == "json":
        payload = _thread_export_payload(target, values, messages, events)
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    else:
        text = _thread_export_markdown(target, messages, events)

    if out_path is None:
        suffix = "json" if fmt == "json" else "md"
        out_path = Path.cwd() / f"askanswer-{target.thread_id[:8]}.{suffix}"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    except Exception as exc:
        render_error(f"写入导出文件失败: {exc}")
        return
    print()
    print(f"  {C.GREEN}✓ 已导出:{C.RESET} {out_path}")
    print()


def handle_import_command(args: str, *, app=None) -> str | None:
    if app is None:
        render_error("/import 只能在已初始化的图上使用")
        return None
    parts = _split_args(args)
    if not parts:
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/import <path.json>{C.RESET}\n")
        return None
    path = Path(parts[0]).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = messages_from_dict(payload.get("messages") or [])
    except Exception as exc:
        render_error(f"导入失败: {exc}")
        return None
    values = dict(payload.get("values") or {})
    values["messages"] = messages
    values["pending_confirmations"] = {}
    values["pending_shell"] = {}
    new_id = str(uuid.uuid4())
    tenant = _current_tenant()
    try:
        _update_state(app, {"configurable": {"thread_id": new_id}}, values)
        meta = payload.get("meta") or {}
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        preview = _latest_human_preview(messages)
        get_persistence().upsert_meta(
            new_id,
            title=f"[imported] {meta.get('title') or preview or path.name}"[:120],
            intent=values.get("intent") or meta.get("last_intent"),
            model_label=meta.get("model_label") or current_model_label(),
            preview=preview,
            message_count=human_count,
            tenant_id=tenant,
        )
        imported_events = get_persistence().import_audit_events(
            payload.get("audit") or [],
            thread_id=new_id,
            tenant_id=tenant,
        )
    except Exception as exc:
        render_error(f"写入导入会话失败: {exc}")
        return None
    print()
    print(
        f"  {C.GREEN}✓ 已导入新会话:{C.RESET} "
        f"{C.CYAN}{new_id[:8]}{C.RESET}  "
        f"{C.DIM}{len(messages)} messages · {imported_events} audit events{C.RESET}"
    )
    print(f"  {C.DIM}已切换到导入的会话。{C.RESET}\n")
    return new_id


def _parse_export_args(args: str, *, current: str) -> tuple[ThreadMeta, str, Path | None] | None:
    parts = _split_args(args)
    if parts is None:
        return None
    target_arg = None
    fmt = "md"
    out_path = None
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "--format" and i + 1 < len(parts):
            fmt = parts[i + 1].lower()
            i += 2
        elif part == "--out" and i + 1 < len(parts):
            out_path = Path(parts[i + 1]).expanduser()
            i += 2
        elif part.startswith("--"):
            print(f"\n  {C.RED}未知参数：{C.RESET}{part}\n")
            return None
        elif target_arg is None:
            target_arg = part
            i += 1
        else:
            print(f"\n  {C.RED}多余参数：{C.RESET}{part}\n")
            return None
    if fmt not in {"md", "json"}:
        print(f"\n  {C.RED}--format 只能是 md 或 json{C.RESET}\n")
        return None
    target = _resolve_thread_or_current(target_arg, current)
    if target is None:
        print(f"\n  {C.RED}找不到匹配的会话：{C.RESET}{target_arg}\n")
        return None
    return target, fmt, out_path


def _thread_export_payload(
    meta: ThreadMeta,
    values: dict,
    messages: list[BaseMessage],
    events: list[AuditEvent],
) -> dict:
    state_values = {
        key: value
        for key, value in values.items()
        if key not in {"messages", "pending_shell", "pending_confirmations"}
    }
    return {
        "version": 1,
        "thread_id": meta.thread_id,
        "exported_at": int(time.time()),
        "meta": asdict(meta),
        "values": state_values,
        "messages": messages_to_dict(messages),
        "audit": [asdict(event) for event in events],
    }


def _thread_export_markdown(
    meta: ThreadMeta,
    messages: list[BaseMessage],
    events: list[AuditEvent],
) -> str:
    title = meta.title or meta.preview or meta.thread_id
    lines = [
        f"# {title}",
        "",
        f"- Thread: `{meta.thread_id}`",
        f"- Exported: {_format_ts(int(time.time()))}",
        f"- Messages: {len(messages)}",
        "",
        "## Conversation",
        "",
    ]
    for message in messages:
        role = getattr(message, "type", type(message).__name__)
        name = getattr(message, "name", None)
        header = f"### {role}" + (f" · {name}" if name else "")
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        if role == "tool" and len(content) > 1200:
            content = content[:1200] + "\n\n...(tool output truncated in markdown export)"
        lines.extend([header, "", content or "_(empty)_", ""])
    if events:
        lines.extend(["## Audit Summary", ""])
        for event in events[:50]:
            detail = event.tool_name or event.model_label or event.args_summary or ""
            lines.append(
                f"- `{_format_ts(event.ts)}` `{event.kind}` "
                f"{_truncate(detail, 90)}"
            )
        lines.append("")
    return "\n".join(lines)


def _latest_human_preview(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip().replace("\n", " ")[:80] or None
    return None
