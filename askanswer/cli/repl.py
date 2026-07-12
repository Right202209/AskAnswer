# 交互式 REPL 主循环 + ``!<cmd>`` 直执行 shell + 输入框边框。
from __future__ import annotations

import uuid

from ..load import current_model_label
from ..mcp import get_manager as _mcp_manager
from ..tools import check_dangerous, execute_shell_command
from ..ui_input import make_session, read_line
from .commands import handle_command
from .render import render_error, tips_block, welcome_box
from .stream import stream_query
from .text import _term_width
from .theme import C, _console


def run_bang_command(command: str) -> None:
    """`!<cmd>` 走绕过 LangGraph 的直执行路径；危险命令仍要二次确认。"""
    command = command.strip()
    if not command:
        # 空命令：打印用法
        print()
        print(f"  {C.DIM}用法：{C.RESET}{C.CYAN}!<shell command>{C.RESET}  "
              f"{C.DIM}例：!ls -la{C.RESET}")
        print()
        return

    # 命中危险模式：必须用户显式确认
    danger = check_dangerous(command)
    if danger:
        print()
        print(f"  {C.RED}⚠ 高风险命令（{danger}）{C.RESET}")
        print(f"    {C.DIM}命令：{C.RESET}{C.CYAN}{command}{C.RESET}")
        try:
            reply = input(f"    {C.ORANGE}仍然执行? (y/N):{C.RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            reply = ""
        if reply not in ("y", "yes"):
            print(f"    {C.DIM}已取消。{C.RESET}")
            print()
            return

    # bang 模式 shell=True：保留管道、重定向等用户自己手敲的语法
    output = execute_shell_command(command, shell=True)

    print()
    for line in output.splitlines():
        print(f"    {line}")
    print()


def _build_status_provider(thread_box: list[str]):
    """构造给 ``ui_input.make_session`` 用的 status 回调。

    ``thread_box`` 是单元素列表，作为对当前 ``thread_id`` 的可变引用
    （``/clear`` / ``/resume`` / ``/delete`` 切换会话时直接改写 ``thread_box[0]``）。
    """
    def get_status() -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = [
            ("thread", (thread_box[0] or "?")[:7]),
            ("model",  current_model_label() or "—"),
        ]
        try:
            servers = _mcp_manager().list_servers()
            if servers:
                items.append(("mcp", str(len(servers))))
        except Exception:
            pass
        return items

    return get_status


def _draw_top_border() -> None:
    """与 prompt_toolkit 输入区配套的视觉上边框；下边框由 ``_draw_bottom_border`` 收尾。"""
    w = _term_width()
    _console.print(f"[muted]╭{'─' * (w - 2)}╮[/]")


def _draw_bottom_border() -> None:
    w = _term_width()
    _console.print(f"[muted]╰{'─' * (w - 2)}╯[/]")


def interactive_loop(app) -> int:
    """REPL 主循环：每轮读一行用户输入，按前缀决定走哪个分支。"""
    # 一个会话用一个 thread_id；用单元素列表包一层，方便 status 回调随时取最新值
    thread_box: list[str] = [str(uuid.uuid4())]
    session = make_session(_build_status_provider(thread_box))

    welcome_box(thread_box[0])
    tips_block()

    while True:
        _draw_top_border()
        try:
            text = read_line(session)
        finally:
            # 输入结束后立刻把下边框补完，无论是正常提交还是 Ctrl-C
            _draw_bottom_border()

        if text is None:
            # Ctrl-D 或 2 秒内连按 Ctrl-C → 退出
            print(f"\n{C.DIM}再见。{C.RESET}")
            return 0
        text = text.strip()
        if not text:
            # 单次 Ctrl-C 或空输入：提示一下再继续
            print(f"  {C.DIM}(已取消；再次 Ctrl-C 退出，或输入 /exit){C.RESET}")
            continue

        # 斜杠命令：走 handle_command 分发
        if text.startswith("/"):
            keep_going, new_id = handle_command(
                text, thread_id=thread_box[0], app=app,
            )
            thread_box[0] = new_id  # /clear、/resume、/delete 可能换 id
            if not keep_going:
                return 0
            continue

        # ! 前缀：直执行 shell
        if text.startswith("!"):
            run_bang_command(text[1:])
            continue

        # 普通输入：交给 LangGraph 处理（stream_query 自己负责答案渲染）
        try:
            stream_query(app, text, thread_box[0])
        except Exception as exc:
            render_error(str(exc))
            continue
