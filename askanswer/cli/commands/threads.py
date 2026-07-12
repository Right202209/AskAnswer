# 会话管理命令：``/threads``（列表）、``/resume``（切换）、``/title``（命名）、``/delete``（删除）。
from __future__ import annotations

import uuid

from rich import box
from rich.table import Table

from ...persistence import get_persistence
from ..render import render_error
from ..text import _format_ts, _truncate
from ..theme import C, _console
from ._common import (
    _current_tenant,
    _has_pending_interrupt,
    _resolve_thread,
    forget_thread,
    remember_threads,
)


def handle_threads_command(args: str, *, current: str) -> None:
    """``/threads [keyword]``：按 updated_at 倒序列出最近 50 条（限当前 tenant）。"""
    keyword = args.strip() or None
    tenant = _current_tenant()
    try:
        threads = get_persistence().list_threads(
            limit=50, query=keyword, tenant_id=tenant
        )
    except Exception as exc:
        render_error(f"读取持久化失败: {exc}")
        return
    remember_threads(threads)

    _console.print()
    if not threads:
        hint = f"无匹配 '{keyword}'" if keyword else "暂无历史会话（先聊几句吧）"
        _console.print(f"  [subtle]（{hint}）[/]")
        _console.print()
        return

    tenant_word = f"  [subtle]· tenant: {tenant}[/]" if tenant else ""
    title_word = f"  [subtle]· 关键词: {keyword}[/]" if keyword else ""
    _console.print(f"  [bold]Threads ({len(threads)})[/]{tenant_word}{title_word}")

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="subtle",
        border_style="muted",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("", width=1, no_wrap=True)
    table.add_column("#", justify="right", style="subtle", no_wrap=True, width=3)
    table.add_column("id", style="info", no_wrap=True, width=8)
    table.add_column("updated", style="subtle", no_wrap=True, width=11)
    table.add_column("intent", style="subtle", no_wrap=True, width=8)
    table.add_column("msgs", justify="right", style="subtle", no_wrap=True, width=4)
    table.add_column("title / preview", no_wrap=True, overflow="ellipsis", ratio=1)

    for i, m in enumerate(threads, 1):
        marker = "[success]●[/]" if m.thread_id == current else " "
        text = (m.title or m.preview or "(空)").strip().replace("\n", " ")
        intent = (m.last_intent or "—")[:8]
        table.add_row(
            marker,
            str(i),
            m.thread_id[:8],
            _format_ts(m.updated_at),
            intent,
            f"{m.message_count}",
            text,
        )
    _console.print(table)
    _console.print(
        "  [subtle]用法：[/]"
        "[info]/resume <序号|id>[/][subtle] 恢复 · [/]"
        "[info]/title <名字>[/][subtle] 命名当前 · [/]"
        "[info]/delete <序号|id>[/][subtle] 删除[/]"
    )
    _console.print()


def handle_resume_command(args: str, *, current: str, app=None) -> str | None:
    """``/resume <序号|id 前缀>``：切换 thread_id 到目标会话。

    返回新的 thread_id；解析失败或用户取消时返回 ``None``。
    """
    if not args.strip():
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/resume <序号|id 前缀>{C.RESET}\n")
        return None

    target = _resolve_thread(args)
    if target is None:
        print(
            f"\n  {C.RED}找不到匹配的会话：{C.RESET}{args}  "
            f"{C.DIM}（先 /threads 看一下序号或 id 前缀）{C.RESET}\n"
        )
        return None

    if target.thread_id == current:
        print(f"\n  {C.DIM}已经在该会话上了：{target.thread_id[:8]}…{C.RESET}\n")
        return None

    # 关键风险：目标会话上次卡在 shell HITL 没 resume；提醒但不阻断
    if _has_pending_interrupt(app, target.thread_id):
        print()
        print(
            f"  {C.GOLD}⚠ 该会话上次中断在确认操作未完成。{C.RESET}\n"
            f"    {C.DIM}下一条问题会作为新一轮开始；挂起的确认会被 LangGraph 保留，"
            f"行为可能怪异。{C.RESET}\n"
            f"    {C.DIM}如需先恢复挂起项，请取消本次切换并直接在原会话回答 y/N。{C.RESET}"
        )
        try:
            reply = input(f"    {C.ORANGE}仍要切换？(y/N):{C.RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            reply = ""
        if reply not in ("y", "yes"):
            print(f"    {C.DIM}已取消。{C.RESET}\n")
            return None

    label = target.title or target.preview or "(无标题)"
    print()
    print(
        f"  {C.GREEN}✓ 已切换到会话:{C.RESET} "
        f"{C.CYAN}{target.thread_id[:8]}{C.RESET}  "
        f"{C.DIM}{_truncate(label, 50)}{C.RESET}"
    )
    print(f"  {C.DIM}下一条问题会接续这段会话的历史。{C.RESET}\n")
    return target.thread_id


def handle_title_command(args: str, *, thread_id: str) -> None:
    """``/title <name>``：给当前会话命名。"""
    title = args.strip()
    if not title:
        try:
            meta = get_persistence().get_meta(thread_id)
        except Exception:
            meta = None
        current = meta.title if meta else None
        print()
        print(f"  {C.BOLD}Title{C.RESET}")
        print(f"   {C.DIM}current:{C.RESET} {current or '(未命名)'}")
        print(f"   {C.DIM}usage:{C.RESET}   {C.CYAN}/title <名字>{C.RESET}")
        print()
        return

    try:
        ok = get_persistence().set_title(thread_id, title)
    except Exception as exc:
        render_error(f"重命名失败: {exc}")
        return

    print()
    if ok:
        print(f"  {C.GREEN}✓ 已命名:{C.RESET} {title}")
    else:
        # 没行被更新：通常是当前会话还没产生任何 final_answer，meta 行尚未写入
        print(
            f"  {C.GOLD}⚠ 当前会话尚未持久化{C.RESET}  "
            f"{C.DIM}（先问一个问题让 thread_meta 写入，再 /title 重命名）{C.RESET}"
        )
    print()


def handle_delete_command(args: str, *, current: str) -> str | None:
    """``/delete <序号|id 前缀>``：删除 thread（含 checkpoints + thread_meta）。

    若删除的是 *当前* thread，返回一个新生成的 thread_id 让 REPL 切换；
    其它情况返回 ``None``。
    """
    if not args.strip():
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/delete <序号|id 前缀>{C.RESET}\n")
        return None

    target = _resolve_thread(args)
    if target is None:
        print(
            f"\n  {C.RED}找不到匹配的会话：{C.RESET}{args}  "
            f"{C.DIM}（先 /threads 看一下序号或 id 前缀）{C.RESET}\n"
        )
        return None

    label = target.title or target.preview or "(无标题)"
    is_current = target.thread_id == current
    print()
    print(f"  {C.RED}⚠  即将删除会话{C.RESET}")
    print(
        f"    {C.DIM}id:{C.RESET}    {C.CYAN}{target.thread_id[:8]}{C.RESET}  "
        f"{C.DIM}{_truncate(label, 50)}{C.RESET}"
    )
    if is_current:
        print(f"    {C.GOLD}这是当前会话；删除后将自动开始新会话。{C.RESET}")
    print(f"    {C.DIM}操作不可撤销（同步清除 checkpoints + thread_meta + audit）。{C.RESET}")
    try:
        reply = input(f"    {C.ORANGE}确认删除? (y/N):{C.RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        reply = ""
    if reply not in ("y", "yes"):
        print(f"    {C.DIM}已取消。{C.RESET}\n")
        return None

    try:
        ok = get_persistence().delete_thread(
            target.thread_id, tenant_id=_current_tenant()
        )
    except Exception as exc:
        render_error(f"删除失败: {exc}")
        return None

    # _LAST_LIST 里这一条已失效，做个简单清理避免 /resume 误中
    forget_thread(target.thread_id)

    print()
    if ok:
        print(f"  {C.GREEN}✓ 已删除:{C.RESET} {target.thread_id[:8]}…")
    else:
        print(f"  {C.GOLD}⚠ thread_meta 中未找到该 ID（可能 checkpoint 已清）{C.RESET}")
    print()

    if is_current:
        new_id = str(uuid.uuid4())
        print(f"  {C.DIM}已开始新会话：{new_id[:8]}…{C.RESET}\n")
        return new_id
    return None
