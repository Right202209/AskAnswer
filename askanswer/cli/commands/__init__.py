# 斜杠命令路由：把 ``/xxx`` 分发到各域命令模块。
#
# 各命令的实现按域拆在同级模块（model / mcp / threads / timetravel / audit / transfer）；
# 本文件只做分发，并把可能换 thread_id 的命令返回值回传给 REPL。
from __future__ import annotations

import os
import uuid

from ..render import help_block, status_block, welcome_box
from ..theme import C
from .audit import handle_audit_command, handle_usage_command
from .mcp import handle_mcp_command
from .model import handle_model_command
from .threads import (
    handle_delete_command,
    handle_resume_command,
    handle_threads_command,
    handle_title_command,
)
from .timetravel import (
    handle_checkpoints_command,
    handle_fork_command,
    handle_jump_command,
    handle_undo_command,
)
from .transfer import handle_export_command, handle_import_command

__all__ = ["handle_command"]


def handle_command(cmd: str, *, thread_id: str, app=None) -> tuple[bool, str]:
    """斜杠命令路由。返回 (是否继续运行, 当前 thread_id)。

    ``app`` 参数：``/resume`` 需要查询 ``app.get_state`` 检测挂起的 interrupt；
    其它命令不强依赖。
    """
    stripped = cmd.strip()
    head, _, tail = stripped.partition(" ")
    head_lc = head.lower()
    tail = tail.strip()

    if head_lc in {"/exit", "/quit", "/q"}:
        print(f"\n{C.DIM}再见。{C.RESET}")
        return False, thread_id
    if head_lc == "/help":
        help_block(tail or None)
    elif head_lc == "/clear":
        # 清屏 + 新建一个 thread_id：等价于一段全新对话。旧线程仍保留在 SqliteSaver 里，
        # 用户可用 /threads 查看 / /resume 恢复 / /delete 删除。
        old_short = thread_id[:8]
        os.system("cls" if os.name == "nt" else "clear")
        thread_id = str(uuid.uuid4())
        welcome_box(thread_id)
        print(
            f"\n  {C.DIM}已开始新会话：{thread_id[:8]}…  "
            f"上一段保留为 {old_short}…（/threads 查看 · /delete 删除）{C.RESET}\n"
        )
    elif head_lc == "/status":
        status_block(thread_id)
    elif head_lc == "/model":
        handle_model_command(tail, thread_id=thread_id)
    elif head_lc == "/mcp":
        handle_mcp_command(tail, thread_id=thread_id)
    elif head_lc == "/threads":
        handle_threads_command(tail, current=thread_id)
    elif head_lc == "/resume":
        new_id = handle_resume_command(tail, current=thread_id, app=app)
        if new_id:
            thread_id = new_id
    elif head_lc == "/title":
        handle_title_command(tail, thread_id=thread_id)
    elif head_lc == "/delete":
        new_id = handle_delete_command(tail, current=thread_id)
        if new_id:
            # 删的是当前 thread：自动开新会话
            thread_id = new_id
    elif head_lc == "/checkpoints":
        handle_checkpoints_command(tail, thread_id=thread_id, app=app)
    elif head_lc == "/undo":
        handle_undo_command(tail, thread_id=thread_id, app=app)
    elif head_lc == "/jump":
        handle_jump_command(tail, thread_id=thread_id, app=app)
    elif head_lc == "/fork":
        new_id = handle_fork_command(tail, current=thread_id, app=app)
        if new_id:
            thread_id = new_id
    elif head_lc == "/audit":
        handle_audit_command(tail, current=thread_id)
    elif head_lc == "/usage":
        handle_usage_command(tail, current=thread_id)
    elif head_lc == "/export":
        handle_export_command(tail, current=thread_id, app=app)
    elif head_lc == "/import":
        new_id = handle_import_command(tail, app=app)
        if new_id:
            thread_id = new_id
    else:
        print(
            f"\n  {C.RED}未知命令：{C.RESET}{stripped}  "
            f"({C.CYAN}/help{C.RESET} 查看可用命令)\n"
        )
    return True, thread_id
