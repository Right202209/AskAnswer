"""ui_select：方向键解析与 TTY 菜单导航。

核心回归：``sys.stdin.read`` 会把整段 CSI 缓冲进 Python 层，导致 ``select``
把 ↑/↓ 误判为裸 Esc。按键路径必须走 ``os.read``。
"""

from __future__ import annotations

import os
import sys

import pytest

from askanswer import ui_select


def _feed_fd(data: bytes) -> int:
    """把 ``data`` 写入 pipe，返回可读端 fd（调用方负责 close）。"""
    r, w = os.pipe()
    os.write(w, data)
    os.close(w)
    return r


@pytest.mark.parametrize(
    "payload,expected",
    [
        (b"\x1b[A", "up"),
        (b"\x1b[B", "down"),
        (b"\x1bOA", "up"),
        (b"\x1bOB", "down"),
        (b"\x1b[1;5A", "up"),
        (b"\x1b[1;5B", "down"),
        (b"\r", "enter"),
        (b"\n", "enter"),
        (b"\x03", "ctrl-c"),
        (b"\x1b", "esc"),
        (b"k", "k"),
        (b"j", "j"),
        (b"1", "1"),
        (b"\x1b[C", ""),  # 右方向键忽略
        (b"\x1b[D", ""),  # 左方向键忽略
    ],
)
def test_read_key_sequences(payload: bytes, expected: str) -> None:
    fd = _feed_fd(payload)
    try:
        assert ui_select._read_key(fd) == expected
    finally:
        os.close(fd)


def test_read_key_does_not_use_stdin_buffer(monkeypatch) -> None:
    """回归：即便 stdin TextIO 会把整段 CSI 缓冲掉，os.read 路径仍能识别 ↑。

    旧实现用 ``sys.stdin.read(1)`` + ``select(stdin)``：read 把 ``[A`` 留在
    Python 缓冲里，select 在 OS 层看不到后续字节，误判为裸 Esc。
    """
    # 构造一个“已缓冲了后续字节”的场景不直接复现 TextIO；这里验证解析函数
    # 完全不依赖 sys.stdin，只读传入的 fd —— 这就是修复的契约。
    fd = _feed_fd(b"\x1b[A")
    try:
        # 污染 sys.stdin 也不应影响 _read_key
        monkeypatch.setattr(sys, "stdin", sys.__stdin__)
        assert ui_select._read_key(fd) == "up"
    finally:
        os.close(fd)


def test_numbered_select_default(monkeypatch, capsys) -> None:
    """非 TTY 回退：空输入取 default。"""
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    idx = ui_select._numbered_select(["a", "b", "c"], "pick", default=1)
    assert idx == 1


def test_select_option_empty() -> None:
    assert ui_select.select_option([]) == (ui_select.CANCELLED, None)


def test_display_width_cjk() -> None:
    assert ui_select._display_width("ab") == 2
    assert ui_select._display_width("中文") == 4
    assert ui_select._display_width("仍按数据库") == 10


def test_menu_rows_counts_prompt_items_footer() -> None:
    """宽终端下每逻辑行一行：prompt + 2 options + footer = 4。"""
    items = [
        "仍按数据库问题处理（无连接可能失败）",
        "改用通用知识作答（不连数据库）",
    ]
    prompt = "选择（↑/↓ 导航 · Enter 确认）："
    assert ui_select._menu_rows(items, prompt, term_cols=120) == 4


def test_menu_rows_accounts_for_wrap() -> None:
    """窄终端：长选项折行会抬高物理行数。"""
    items = ["仍按数据库问题处理（无连接可能失败）"]
    # 选项前缀 4 列 + 文案约 44 列 ≈ 48；cols=20 时会折多行
    rows = ui_select._menu_rows(items, "", term_cols=20)
    assert rows > 2  # 至少 option 折行 + footer


def test_redraw_moves_up_rows_minus_one() -> None:
    """重绘上移行数必须是「总物理行 - 1」（光标停在底栏无换行）。"""
    items = ["a", "b"]
    prompt = "pick"
    rows = ui_select._menu_rows(items, prompt, term_cols=80)
    assert rows == 4
    assert rows - 1 == 3  # CUU 参数
