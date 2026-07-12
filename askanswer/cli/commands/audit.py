# 审计与用量命令：``/audit``（事件流）、``/usage``（模型/工具用量与费用估算）。
from __future__ import annotations

from rich import box
from rich.table import Table

from ...persistence import AuditEvent, ThreadMeta, get_persistence
from ...pricing import estimate_cost_usd, format_cost
from ..render import render_error
from ..text import _format_ts, _truncate
from ..theme import C, _console
from ._common import (
    _current_tenant,
    _parse_nonnegative_int,
    _resolve_thread_or_current,
    _split_args,
)


def handle_audit_command(args: str, *, current: str) -> None:
    parts = _split_args(args)
    if parts is None:
        return
    target_arg = None
    kind = None
    limit = 30
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "--kind" and i + 1 < len(parts):
            kind = parts[i + 1]
            i += 2
        elif part == "--limit" and i + 1 < len(parts):
            parsed = _parse_nonnegative_int(parts[i + 1])
            if parsed is None or parsed <= 0:
                print(f"\n  {C.RED}--limit 必须是正整数{C.RESET}\n")
                return
            limit = parsed
            i += 2
        elif part.startswith("--"):
            print(f"\n  {C.RED}未知参数：{C.RESET}{part}\n")
            return
        elif target_arg is None:
            target_arg = part
            i += 1
        else:
            print(f"\n  {C.RED}多余参数：{C.RESET}{part}\n")
            return
    target = _resolve_thread_or_current(target_arg, current)
    if target is None:
        print(f"\n  {C.RED}找不到匹配的会话：{C.RESET}{target_arg}\n")
        return
    try:
        events = get_persistence().list_audit_events(
            thread_id=target.thread_id,
            kind=kind,
            limit=limit,
            tenant_id=_current_tenant(),
        )
    except Exception as exc:
        render_error(f"读取审计失败: {exc}")
        return
    _print_audit_events(events, target)


def handle_usage_command(args: str, *, current: str) -> None:
    parts = _split_args(args)
    if parts is None:
        return
    days = 7
    thread = None
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "--days" and i + 1 < len(parts):
            parsed = _parse_nonnegative_int(parts[i + 1])
            if parsed is None:
                print(f"\n  {C.RED}--days 必须是整数{C.RESET}\n")
                return
            days = parsed
            i += 2
        elif part == "--thread" and i + 1 < len(parts):
            thread = _resolve_thread_or_current(parts[i + 1], current)
            if thread is None:
                print(f"\n  {C.RED}找不到匹配的会话：{C.RESET}{parts[i + 1]}\n")
                return
            i += 2
        else:
            print(f"\n  {C.RED}未知参数：{C.RESET}{part}\n")
            return
    try:
        summary = get_persistence().usage_summary(
            thread_id=thread.thread_id if thread else None,
            days=days,
            tenant_id=_current_tenant(),
        )
    except Exception as exc:
        render_error(f"读取 usage 失败: {exc}")
        return
    _print_usage(summary, days=days, thread=thread)


def _print_audit_events(events: list[AuditEvent], target: ThreadMeta) -> None:
    _console.print()
    label = target.title or target.preview or target.thread_id[:8]
    _console.print(
        f"  [bold]Audit[/]  [subtle]{_truncate(label, 48)}[/]"
    )
    if not events:
        _console.print("  [subtle]（暂无审计事件）[/]")
        _console.print()
        return

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="subtle",
        border_style="muted",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("time", style="subtle", no_wrap=True, width=11)
    table.add_column("kind", style="info", no_wrap=True, width=13)
    table.add_column("target", style="subtle", no_wrap=True, width=18, overflow="ellipsis")
    table.add_column("tokens", style="subtle", justify="right", no_wrap=True, width=16)
    table.add_column("detail", no_wrap=True, overflow="ellipsis", ratio=1)

    for event in events:
        target_text = event.tool_name or event.model_label or ""
        if event.input_tokens is not None or event.output_tokens is not None:
            tokens = f"in={event.input_tokens or 0} out={event.output_tokens or 0}"
        else:
            tokens = ""
        if event.error:
            detail = f"[danger]error:[/] {event.error}"
        else:
            detail = event.args_summary or ""
        table.add_row(
            _format_ts(event.ts),
            event.kind,
            target_text,
            tokens,
            detail,
        )
    _console.print(table)
    _console.print()


def _print_usage(summary: dict, *, days: int, thread: ThreadMeta | None) -> None:
    _console.print()
    scope = f"thread {thread.thread_id[:8]}" if thread else "all threads"
    window = "all time" if days == 0 else f"{days}d"
    _console.print(f"  [bold]Usage[/]  [subtle]{scope} · {window}[/]")

    models = summary.get("models") or []
    if models:
        m_table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="subtle",
            border_style="muted",
            padding=(0, 1),
            expand=False,
            title="Models",
            title_style="subtle",
            title_justify="left",
        )
        m_table.add_column("model", style="info", no_wrap=True, max_width=28, overflow="ellipsis")
        m_table.add_column("calls", justify="right", style="subtle", no_wrap=True)
        m_table.add_column("in", justify="right", style="subtle", no_wrap=True)
        m_table.add_column("out", justify="right", style="subtle", no_wrap=True)
        m_table.add_column("cost", justify="right", no_wrap=True)
        for row in models:
            cost = estimate_cost_usd(
                row.get("model_label"),
                row.get("input_tokens"),
                row.get("output_tokens"),
            )
            m_table.add_row(
                row.get("model_label") or "unknown",
                f"{row.get('calls', 0)}",
                f"{row.get('input_tokens', 0)}",
                f"{row.get('output_tokens', 0)}",
                format_cost(cost),
            )
        _console.print(m_table)
    else:
        _console.print("  [subtle]Models: no LLM usage recorded[/]")

    tools = summary.get("tools") or []
    if tools:
        t_table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="subtle",
            border_style="muted",
            padding=(0, 1),
            expand=False,
            title="Tools / events",
            title_style="subtle",
            title_justify="left",
        )
        t_table.add_column("name", style="info", no_wrap=True, max_width=28, overflow="ellipsis")
        t_table.add_column("calls", justify="right", style="subtle", no_wrap=True)
        t_table.add_column("chars", justify="right", style="subtle", no_wrap=True)
        t_table.add_column("errors", justify="right", no_wrap=True)
        for row in tools:
            err = row.get("errors", 0)
            err_text = f"[danger]{err}[/]" if err else "[subtle]0[/]"
            t_table.add_row(
                row.get("name") or "unknown",
                f"{row.get('calls', 0)}",
                f"{row.get('result_size', 0)}",
                err_text,
            )
        _console.print(t_table)
    _console.print()
