# 时间旅行命令：``/checkpoints``（列出）、``/undo``（回退 + 命名还原点）、``/jump``、``/fork``。
from __future__ import annotations

from rich import box
from rich.table import Table

from ...persistence import get_persistence
from ...timetravel import (
    find_checkpoint_index_by_id,
    fork_thread,
    list_checkpoints,
    rewind_to,
)
from ..render import render_error
from ..text import _format_ts
from ..theme import C, _console
from ._common import _parse_nonnegative_int, _split_args


def handle_checkpoints_command(args: str, *, thread_id: str, app=None) -> None:
    if app is None:
        render_error("/checkpoints 只能在已初始化的图上使用")
        return
    try:
        checkpoints = list_checkpoints(app, thread_id)
    except Exception as exc:
        render_error(f"读取 checkpoints 失败: {exc}")
        return

    _console.print()
    if not checkpoints:
        _console.print("  [subtle]（当前会话暂无 checkpoints）[/]")
        _console.print()
        return
    _console.print(f"  [bold]Checkpoints ({len(checkpoints)})[/]")

    # checkpoint_id → label 映射，用于在表格里展示命名的还原点。
    try:
        labels = {
            item["checkpoint_id"]: item["label"]
            for item in get_persistence().list_checkpoint_labels(thread_id)
        }
    except Exception:
        labels = {}

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="subtle",
        border_style="muted",
        padding=(0, 1),
        expand=False,
    )
    table.add_column("#", justify="right", style="subtle", no_wrap=True)
    table.add_column("node", style="info", no_wrap=True)
    table.add_column("created", style="subtle", no_wrap=True)
    table.add_column("msgs", justify="right", style="subtle", no_wrap=True)
    table.add_column("step", style="subtle", no_wrap=True)
    table.add_column("label", style="brand", no_wrap=True)
    table.add_column("flags", no_wrap=True)

    for cp in checkpoints:
        flags = []
        if cp.index == 0:
            flags.append("[success]latest[/]")
        if cp.pending_confirm:
            flags.append("[warning]pending-confirm[/]")
        table.add_row(
            str(cp.index),
            cp.node,
            _format_ts(cp.created_at),
            str(cp.message_count),
            cp.step,
            labels.get(cp.checkpoint_id, ""),
            " ".join(flags),
        )
    _console.print(table)
    _console.print(
        "  [subtle]用法：[/]"
        "[info]/undo [n] [--label 名称][/][subtle] 回退并可命名还原点 · [/]"
        "[info]/jump <index>[/][subtle] 显式跳转 · [/]"
        "[info]/fork [index][/][subtle] 分叉新会话[/]"
    )
    _console.print()


def handle_undo_command(args: str, *, thread_id: str, app=None) -> None:
    parts = _split_args(args)
    if parts is None:
        return
    parsed = _parse_undo_args(parts)
    if parsed is None:
        return
    index, ckpt_label = parsed
    # 纯 /undo 无参：沿用旧行为回退 1 步；带 --label 但无序号则走 label 反查。
    if index is None and not ckpt_label:
        index = 1
    _rewind_command(app, thread_id, index, cmd="/undo", checkpoint_label=ckpt_label)


def _parse_undo_args(parts: list[str]) -> tuple[int | None, str | None] | None:
    """解析 ``/undo`` 参数：``[n] [--label NAME]``。

    返回 ``(index, label)``；index 为 None 表示未显式给序号（可能走 label 反查）。
    解析出错时打印用法并返回 None。
    """
    index: int | None = None
    label: str | None = None
    i = 0
    usage = f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/undo [n] [--label 名称]{C.RESET}\n"
    while i < len(parts):
        part = parts[i]
        if part == "--label":
            if i + 1 >= len(parts):
                print(usage)
                return None
            label = parts[i + 1].strip() or None
            i += 2
        elif part.startswith("--"):
            print(f"\n  {C.RED}未知参数：{C.RESET}{part}\n")
            return None
        elif index is None:
            parsed = _parse_nonnegative_int(part)
            if parsed is None or parsed < 1:
                print(usage)
                return None
            index = parsed
            i += 1
        else:
            print(f"\n  {C.RED}多余参数：{C.RESET}{part}\n")
            return None
    return index, label


def handle_jump_command(args: str, *, thread_id: str, app=None) -> None:
    index = _parse_nonnegative_int(args.strip())
    if index is None:
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/jump <checkpoint-index>{C.RESET}\n")
        return
    _rewind_command(app, thread_id, index, cmd="/jump")


def _rewind_command(
    app,
    thread_id: str,
    index: int | None,
    *,
    cmd: str,
    checkpoint_label: str | None = None,
) -> None:
    if app is None:
        render_error(f"{cmd} 只能在已初始化的图上使用")
        return
    pm = get_persistence()
    # 显式序号 + --label：回滚后把 label 记到目标 checkpoint 上（命名还原点）。
    # 仅 --label 无序号：按 label 反查历史里的 checkpoint 再回滚。
    naming = index is not None and bool(checkpoint_label)
    if index is None:
        index = _resolve_label_index(app, pm, thread_id, checkpoint_label, cmd=cmd)
        if index is None:
            return
    try:
        result = rewind_to(app, thread_id, index)
    except Exception as exc:
        render_error(f"{cmd} 失败: {exc}")
        return
    target = result.target
    if naming and target.checkpoint_id:
        pm.set_checkpoint_label(thread_id, target.checkpoint_id, checkpoint_label)
    label_note = f"  {C.DIM}· label={checkpoint_label}{C.RESET}" if checkpoint_label else ""
    print()
    print(
        f"  {C.GREEN}✓ 已回到 checkpoint:{C.RESET} "
        f"{C.CYAN}#{target.index}{C.RESET}  "
        f"{C.DIM}{target.node} · {target.message_count} msgs · step={target.step}{C.RESET}"
        f"{label_note}"
    )
    print(
        f"  {C.DIM}本次回滚影响 {result.affected_messages} 条消息；"
        f"下一条问题会基于该快照继续。{C.RESET}\n"
    )


def _resolve_label_index(app, pm, thread_id: str, label: str | None, *, cmd: str) -> int | None:
    """把 checkpoint label 反查成当前 history 里的序号；失败时打印错误并返回 None。"""
    if not label:
        render_error(f"{cmd} 需要序号或 --label 名称")
        return None
    checkpoint_id = pm.resolve_checkpoint_label(thread_id, label)
    resolved = find_checkpoint_index_by_id(app, thread_id, checkpoint_id) if checkpoint_id else None
    if resolved is None:
        render_error(f"未找到 label 对应的 checkpoint：{label}")
        return None
    return resolved


def handle_fork_command(args: str, *, current: str, app=None) -> str | None:
    if app is None:
        render_error("/fork 只能在已初始化的图上使用")
        return None
    raw = args.strip()
    index = 0 if not raw else _parse_nonnegative_int(raw)
    if index is None:
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/fork [checkpoint-index]{C.RESET}\n")
        return None
    try:
        new_id = fork_thread(app, current, get_persistence(), index=index)
    except Exception as exc:
        render_error(f"分叉失败: {exc}")
        return None
    print()
    print(
        f"  {C.GREEN}✓ 已分叉新会话:{C.RESET} "
        f"{C.CYAN}{new_id[:8]}{C.RESET}  {C.DIM}来源 checkpoint #{index}{C.RESET}"
    )
    print(f"  {C.DIM}已切换到新会话；旧会话保持不变。{C.RESET}\n")
    return new_id
