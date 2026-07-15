"""终端上下选项选择器：支持 ↑/↓/k/j 导航、Enter 确认、Esc/Ctrl-C 取消。

复用场景：
- HITL shell 命令确认（``cli._prompt_shell_confirmation``）
- Helix 苏格拉底访谈（``helix.nodes.interview_node``）

设计取舍：
- 仅依赖 ``termios`` + ``tty`` + ``select`` + ``os.read``，避免 prompt_toolkit
  这种重依赖；
- 按键一律走 ``os.read(fd, 1)``：``sys.stdin.read(1)`` 会经 TextIO 缓冲把整段
  CSI（如 ``\\x1b[A``）一口气吞进 Python 层，随后 ``select(stdin)`` 在 OS 层
  看不到后续字节，把方向键误判成裸 Esc 取消 —— 这就是 ↑/↓ 导航失效的根因；
- 重绘用「上移 N 行 + 清到屏尾」而不是 ``ESC[s``/``ESC[u``：保存/恢复光标槽
  会被 rich / prompt_toolkit / 滚动污染，一按方向键就整段菜单叠着往下打；
- ``select_option(...)`` 在非 TTY（如管道/CI）下退化为带编号的文本选择，确保
  非交互流程不会卡死；
- 可选 ``free_input_label`` 在选项末尾追加“其他（手动输入）”，命中时弹出文本
  输入框 —— 这就是“选项 + 用户输入”二合一形态。
"""
from __future__ import annotations

import os
import select as _select
import shutil
import sys
import unicodedata

# ── ANSI codes（重复一份避免与 cli.C 形成相互导入） ─────────────────
_ORANGE = "\033[38;5;214m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GOLD = "\033[38;5;178m"
_GREEN = "\033[38;5;114m"
_RESET = "\033[0m"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_DOWN = "\033[J"  # 从光标清到屏尾，吃掉上次重绘残留
_CLEAR_LINE = "\033[K"

# 取消（Esc / Ctrl-C）的统一返回值
CANCELLED = -1

# select 等待 ESC 后续字节的超时；裸 Esc 与方向键 CSI 的分界。
_ESC_FOLLOW_TIMEOUT = 0.05

# 选项行前缀显示宽度：``  ❯ `` / ``    `` 都按 4 列算（❯ 按 1 列）
_OPTION_PREFIX_COLS = 4
# 提示行 / 底栏左侧缩进 ``  ``
_INDENT_COLS = 2


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

def _read_byte(fd: int) -> bytes:
    """从 fd 无缓冲读 1 字节；EOF / 错误返回空 bytes。"""
    try:
        return os.read(fd, 1)
    except OSError:
        return b""


def _byte_ready(fd: int, timeout: float = _ESC_FOLLOW_TIMEOUT) -> bool:
    """``select`` 探测 fd 上是否还有可读字节（不经过 Python IO 缓冲）。"""
    try:
        rlist, _, _ = _select.select([fd], [], [], timeout)
    except (ValueError, OSError):
        return False
    return bool(rlist)


def _read_key(fd: int) -> str:
    """读一个逻辑按键，返回 ``up`` / ``down`` / ``enter`` / ``esc`` / ``ctrl-c`` /
    单个字符，或空串（忽略的转义序列 / EOF）。

    方向键同时识别：
    - CSI 常规模式 ``ESC [ A/B``
    - SS3 应用模式 ``ESC O A/B``（部分终端 / 全屏程序退出后残留）
    - 带修饰的 CSI（如 ``ESC [ 1 ; 5 A``）—— 仍按最终字节 A/B 处理
    """
    b0 = _read_byte(fd)
    if not b0:
        return "esc"
    if b0 == b"\x03":
        return "ctrl-c"
    if b0 in (b"\r", b"\n"):
        return "enter"
    if b0 != b"\x1b":
        # 普通可打印 / 控制字符：按 latin-1 1:1 映射，避免误触 UTF-8 多字节
        return b0.decode("latin-1")

    # ESC：等后续字节；超时 → 裸 Esc 取消
    if not _byte_ready(fd):
        return "esc"

    b1 = _read_byte(fd)
    if not b1:
        return "esc"

    if b1 == b"[":
        # CSI：吞到 final byte（0x40–0x7E），按末字节判断 ↑/↓
        final = b""
        while _byte_ready(fd):
            c = _read_byte(fd)
            if not c:
                break
            final = c
            if 0x40 <= c[0] <= 0x7E:
                break
        if final == b"A":
            return "up"
        if final == b"B":
            return "down"
        return ""

    if b1 == b"O":
        # SS3 应用光标键
        if not _byte_ready(fd):
            return ""
        b2 = _read_byte(fd)
        if b2 == b"A":
            return "up"
        if b2 == b"B":
            return "down"
        return ""

    # 其它 ESC 前缀（如 Alt+key）忽略
    return ""


def _display_width(text: str) -> int:
    """终端显示列宽（CJK 全宽按 2；组合音标不计）。不含 ANSI。"""
    width = 0
    for ch in text:
        if unicodedata.category(ch) in ("Mn", "Me", "Cf"):
            continue
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def _physical_rows(cols: int, *segments: int) -> int:
    """若干逻辑行在 ``cols`` 宽终端上占几行物理行（折行累加）。"""
    if cols <= 0:
        cols = 80
    total = 0
    for width in segments:
        total += max(1, (max(0, width) + cols - 1) // cols)
    return total


def _menu_rows(items: list[str], prompt: str, term_cols: int) -> int:
    """菜单整体占用的物理行数（与 ``_draw_menu`` 输出一致，含底栏）。"""
    segments: list[int] = []
    if prompt:
        segments.append(_INDENT_COLS + _display_width(prompt))
    for label in items:
        segments.append(_OPTION_PREFIX_COLS + _display_width(label))
    # 底栏：``  ↑/↓ 导航 · Enter 确认 · Esc 取消``
    footer = "↑/↓ 导航 · Enter 确认 · Esc 取消"
    segments.append(_INDENT_COLS + _display_width(footer))
    return _physical_rows(term_cols, *segments)


def _draw_menu(items: list[str], prompt: str, cursor: int) -> None:
    """把菜单写到当前光标位置；最后停在底栏行尾（无尾随换行）。"""
    if prompt:
        sys.stdout.write(f"  {_BOLD}{prompt}{_RESET}{_CLEAR_LINE}\n")
    for i, label in enumerate(items):
        if i == cursor:
            sys.stdout.write(
                f"  {_ORANGE}❯{_RESET} {_BOLD}{label}{_RESET}{_CLEAR_LINE}\n"
            )
        else:
            sys.stdout.write(f"    {_DIM}{label}{_RESET}{_CLEAR_LINE}\n")
    sys.stdout.write(
        f"  {_DIM}↑/↓ 导航 · Enter 确认 · Esc 取消{_RESET}{_CLEAR_LINE}"
    )
    sys.stdout.flush()


def _arrow_select(items: list[str], prompt: str, default: int) -> int:
    """cbreak 模式下读取按键并实时重绘菜单；返回选中索引或 CANCELLED。

    重绘策略：记录上次菜单占用的物理行数，下次先 ``ESC[<n>A`` 回到顶部，
    再 ``ESC[J`` 清到屏尾后重画。避免 ``ESC[s``/``ESC[u`` 保存槽被其它
    组件覆盖后整段菜单向下堆叠。
    """
    import termios
    import tty

    cursor = max(0, min(default, len(items) - 1))
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    drawn_rows = 0

    def term_cols() -> int:
        try:
            return max(1, shutil.get_terminal_size(fallback=(80, 24)).columns)
        except Exception:
            return 80

    def render() -> None:
        nonlocal drawn_rows
        cols = term_cols()
        if drawn_rows > 0:
            # 光标在底栏行尾：上移 (总物理行 - 1) 回到菜单第一行行首，
            # 再清到屏尾，避免旧折行 / 更长文案残留。
            up = drawn_rows - 1
            if up > 0:
                sys.stdout.write(f"\033[{up}A")
            sys.stdout.write(f"\r{_CLEAR_DOWN}")
        _draw_menu(items, prompt, cursor)
        drawn_rows = max(1, _menu_rows(items, prompt, cols))

    def finish() -> None:
        sys.stdout.write(f"\n{_SHOW_CURSOR}")
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        sys.stdout.write(_HIDE_CURSOR)
        sys.stdout.flush()
        render()
        while True:
            key = _read_key(fd)
            if key == "up" or key == "k":
                cursor = (cursor - 1) % len(items)
                render()
            elif key == "down" or key == "j":
                cursor = (cursor + 1) % len(items)
                render()
            elif key == "enter":
                finish()
                return cursor
            elif key == "esc":
                finish()
                return CANCELLED
            elif key == "ctrl-c":
                raise KeyboardInterrupt
            elif key and key.isdigit():
                # 1-9 直接跳到对应项
                pick = int(key) - 1
                if 0 <= pick < len(items):
                    cursor = pick
                    render()
            # 空串 / 其它键：忽略
    except KeyboardInterrupt:
        finish()
        return CANCELLED
    finally:
        # 异常路径也要恢复光标可见 + termios
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()
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
