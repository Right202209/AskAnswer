# ``/edit <path>``：用系统编辑器（nano/vim 等）打开文件。
from __future__ import annotations

import os
import shlex
import shutil
import sys

from ...tools import execute_shell_command
from ..theme import C

# 解析顺序：显式配置 → 常见环境变量 → 本机已安装的编辑器
_EDITOR_ENV_KEYS = ("ASKANSWER_EDITOR", "VISUAL", "EDITOR")
_EDITOR_FALLBACKS = ("nano", "vim", "nvim", "vi", "micro", "emacs")


def resolve_editor() -> list[str] | None:
    """解析要启动的编辑器 argv（不含目标路径）。

    优先 ``ASKANSWER_EDITOR`` / ``VISUAL`` / ``EDITOR``（可含参数，如
    ``code --wait``），否则在 PATH 里找 nano → vim → nvim → vi → …
    """
    for key in _EDITOR_ENV_KEYS:
        raw = (os.environ.get(key) or "").strip()
        if not raw:
            continue
        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = [raw]
        if parts:
            return parts
    for name in _EDITOR_FALLBACKS:
        if shutil.which(name):
            return [name]
    return None


def handle_edit_command(path_arg: str) -> None:
    """``/edit <path>``：在真实 TTY 里拉起编辑器修改文件。"""
    path_arg = (path_arg or "").strip()
    if not path_arg:
        print()
        print(f"  {C.DIM}用法：{C.RESET}{C.CYAN}/edit <path>{C.RESET}")
        print(
            f"  {C.DIM}编辑器：{C.RESET}"
            f"$ASKANSWER_EDITOR → $VISUAL → $EDITOR → nano/vim/…"
        )
        print(f"  {C.DIM}示例：{C.RESET}{C.CYAN}/edit ./README.md{C.RESET}")
        print()
        return

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print()
        print(f"  {C.RED}当前不是交互式终端，无法启动编辑器。{C.RESET}")
        print()
        return

    editor = resolve_editor()
    if editor is None:
        print()
        print(
            f"  {C.RED}未找到可用编辑器。{C.RESET} "
            f"请安装 nano/vim，或设置 "
            f"{C.CYAN}$EDITOR{C.RESET} / {C.CYAN}$ASKANSWER_EDITOR{C.RESET}。"
        )
        print()
        return

    path = os.path.expanduser(path_arg)
    # 拼成一条 shell 安全的命令串，交给 tty 模式执行
    argv = editor + [path]
    command = " ".join(shlex.quote(a) for a in argv)
    editor_label = editor[0]

    print()
    print(
        f"  {C.DIM}打开编辑器{C.RESET} "
        f"{C.CYAN}{editor_label}{C.RESET} "
        f"{C.DIM}→{C.RESET} {path}"
    )
    print()

    result = execute_shell_command(command, shell=False, tty=True)
    # 只回显退出状态行（内容已在 TTY 上交互完成）
    for line in result.splitlines():
        if line.startswith("返回码：") or line.startswith("命令未找到"):
            print(f"  {C.DIM}{line}{C.RESET}")
            break
    print()
