"""终端上下选项选择器：支持 ↑/↓/k/j 导航、Enter 确认、Esc/Ctrl-C 取消。

复用场景：
- HITL shell 命令确认（``cli._prompt_shell_confirmation``）
- Helix 苏格拉底访谈（``helix.nodes.interview_node``）

设计取舍：
- 仅依赖 ``termios`` + ``tty`` + ``select``，避免 prompt_toolkit 这种重依赖；
- ``select_option(...)`` 在非 TTY（如管道/CI）下退化为带编号的文本选择，确保
  非交互流程不会卡死；
- 可选 ``free_input_label`` 在选项末尾追加“其他（手动输入）”，命中时弹出文本
  输入框 —— 这就是“选项 + 用户输入”二合一形态。
"""
from __future__ import annotations

import select as _select
import sys


# ── ANSI codes（重复一份避免与 cli.C 形成相互导入） ─────────────────
_ORANGE = "\033[38;5;214m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GOLD = "\033[38;5;178m"
_GREEN = "\033[38;5;114m"
_RESET = "\033[0m"

# 取消（Esc / Ctrl-C）的统一返回值
CANCELLED = -1


def is_interactive() -> bool:
    """stdin 与 stdout 都连着 TTY 才算交互模式 —— 任何一端被重定向都 fallback。"""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def select_option(
    options: list[str],
    *,
    prompt: str = "",
    default: int = 0,
    free_input_label: str | None = None,
    free_input_prompt: str = "请输入：",
) -> tuple[int, str | None]:
    """弹出一个上下导航的选项菜单。

    参数:
        options: 候选项文本，按渲染顺序排列。
        prompt: 菜单上方的引导语，可空。
        default: 初始光标位置（0-based）。
        free_input_label: 若给出，会作为最后一个选项追加；命中后会弹文本输入框。
        free_input_prompt: 文本输入框的提示语。

    返回:
        ``(index, free_text)``。
        - 选了普通项时 ``free_text`` 为 ``None``；
        - 选了“其他（手动输入）”时 ``free_text`` 为用户键入的字符串（可能为空）；
        - 取消（Esc / Ctrl-C）时 ``index == CANCELLED``。
    """
    items = list(options)
    free_idx = -2  # 一个不可能等于真实 index 的值
    if free_input_label:
        items.append(free_input_label)
        free_idx = len(items) - 1

    if not items:
        return CANCELLED, None

    if is_interactive():
        idx = _arrow_select(items, prompt, default)
    else:
        idx = _numbered_select(items, prompt, default)

    if idx == free_idx:
        return idx, _read_free_input(free_input_prompt)
    return idx, None


# ── interactive arrow-key implementation ──────────────────────────

def _arrow_select(items: list[str], prompt: str, default: int) -> int:
    """cbreak 模式下读取按键并实时重绘菜单；返回选中索引或 CANCELLED。"""
    import termios
    import tty

    cursor = max(0, min(default, len(items) - 1))
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    # 用 ESC[s / ESC[u 保存与还原光标位置；只在第一次渲染前保存一次。
    sys.stdout.write("\033[s")
    sys.stdout.flush()

    def render() -> None:
        sys.stdout.write("\033[u")  # 回到保存位置
        if prompt:
            sys.stdout.write(f"  {_BOLD}{prompt}{_RESET}\033[K\n")
        for i, label in enumerate(items):
            if i == cursor:
                sys.stdout.write(
                    f"  {_ORANGE}❯{_RESET} {_BOLD}{label}{_RESET}\033[K\n"
                )
            else:
                sys.stdout.write(f"    {_DIM}{label}{_RESET}\033[K\n")
        sys.stdout.write(
            f"  {_DIM}↑/↓ 导航 · Enter 确认 · Esc 取消{_RESET}\033[K"
        )
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        render()
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # ESC 后通常是方向键序列；用 select 探一下还有没有后续字节。
                rlist, _, _ = _select.select([sys.stdin], [], [], 0.05)
                if not rlist:
                    return CANCELLED  # 单纯按了 Esc → 取消
                seq = sys.stdin.read(2)
                if seq == "[A":  # ↑
                    cursor = (cursor - 1) % len(items)
                    render()
                elif seq == "[B":  # ↓
                    cursor = (cursor + 1) % len(items)
                    render()
                # 其它转义序列（左/右/F1…）忽略
            elif ch in ("\r", "\n"):
                # 选定后写一行换行，让后续 print 不覆盖菜单底部
                sys.stdout.write("\n")
                sys.stdout.flush()
                return cursor
            elif ch == "k":
                cursor = (cursor - 1) % len(items)
                render()
            elif ch == "j":
                cursor = (cursor + 1) % len(items)
                render()
            elif ch.isdigit():
                # 1-9 直接跳到对应项
                pick = int(ch) - 1
                if 0 <= pick < len(items):
                    cursor = pick
                    render()
            elif ch == "\x03":  # Ctrl-C
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return CANCELLED
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── non-TTY fallback ──────────────────────────────────────────────

def _numbered_select(items: list[str], prompt: str, default: int) -> int:
    """无 TTY 时退化为 1/2/3 数字输入。"""
    if prompt:
        print(f"  {prompt}")
    for i, label in enumerate(items, 1):
        marker = "*" if i == default + 1 else " "
        print(f"   {marker} {i}. {label}")
    try:
        raw = input(f"  选择 1-{len(items)}（回车默认 {default + 1}）：").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return CANCELLED
    if not raw:
        return default
    try:
        n = int(raw)
        if 1 <= n <= len(items):
            return n - 1
    except ValueError:
        pass
    return default


def _read_free_input(prompt: str) -> str:
    """命中“其他（手动输入）”时读一行自由文本。"""
    try:
        return input(f"  {_GOLD}{prompt}{_RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


__all__ = ["CANCELLED", "is_interactive", "select_option"]
