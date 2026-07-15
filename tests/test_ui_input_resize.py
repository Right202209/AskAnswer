"""输入区 resize reflow：erase 行数、continuation 同宽、紧凑底栏。"""

from __future__ import annotations

from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.utils import get_cwidth

from askanswer.ui_input import (
    _build_bottom_toolbar,
    _prompt_continuation,
    _prompt_message,
    _reflow_erase_up_rows,
)


def test_reflow_erase_up_rows_no_change_when_same_or_wider() -> None:
    # 同宽 / 变宽：不放大，按光标相对行上移即可
    assert _reflow_erase_up_rows(3, last_columns=80, new_columns=80, new_rows=40) == 3
    assert _reflow_erase_up_rows(3, last_columns=40, new_columns=80, new_rows=40) == 3
    assert _reflow_erase_up_rows(0, last_columns=80, new_columns=100, new_rows=40) == 0


def test_reflow_erase_up_rows_scales_on_narrow() -> None:
    # 80 → 40：factor=2；cursor_y=3 → (3+1)*2 + 2 = 10
    up = _reflow_erase_up_rows(3, last_columns=80, new_columns=40, new_rows=40)
    assert up == 10

    # 120 → 40：factor=3；cursor_y=2 → (2+1)*3 + 2 = 11
    up = _reflow_erase_up_rows(2, last_columns=120, new_columns=40, new_rows=40)
    assert up == 11

    # 光标在首行也会 reflow：y=0 → 1*2 + 2 = 4
    assert _reflow_erase_up_rows(0, last_columns=80, new_columns=40, new_rows=40) == 4


def test_reflow_erase_up_rows_unknown_last_columns_pads() -> None:
    # 丢失 last_columns 时多擦 3 行，避免连续 resize 叠字
    assert _reflow_erase_up_rows(2, last_columns=None, new_columns=40, new_rows=40) == 5


def test_reflow_erase_up_rows_capped_by_terminal_rows() -> None:
    up = _reflow_erase_up_rows(5, last_columns=200, new_columns=20, new_rows=10)
    # factor=10 → (5+1)*10 + 2 = 62，封顶 rows-1=9
    assert up == 9


def test_prompt_continuation_matches_main_prompt_width() -> None:
    main_w = get_cwidth(fragment_list_to_text(_prompt_message()))
    # 硬换行（wrap_count=0）与软折（wrap_count>0）都必须与主提示同宽
    hard = _prompt_continuation(main_w, line_number=1, wrap_count=0)
    soft = _prompt_continuation(main_w, line_number=0, wrap_count=1)
    assert get_cwidth(fragment_list_to_text(hard)) == main_w
    assert get_cwidth(fragment_list_to_text(soft)) == main_w
    # 若 PTK 传入更大 width，右侧补齐
    padded = _prompt_continuation(main_w + 3, line_number=1, wrap_count=0)
    assert get_cwidth(fragment_list_to_text(padded)) == main_w + 3


def test_prompt_continuation_soft_vs_hard_suffix() -> None:
    hard_text = fragment_list_to_text(_prompt_continuation(4, 1, 0))
    soft_text = fragment_list_to_text(_prompt_continuation(4, 0, 2))
    assert "·" in hard_text
    assert "·" not in soft_text


def test_bottom_toolbar_never_pads_to_full_width() -> None:
    parts = _build_bottom_toolbar(
        [("model", "openai:gpt"), ("mcp", "2"), ("thread", "abc1234")],
        cols=80,
    )
    text = fragment_list_to_text(parts)
    # 紧凑拼接，绝不能用空格铺到全宽（那会触发 reflow 残影）
    assert len(text) < 80
    assert "model:" in text
    assert "mcp:" in text
    assert "thread:" in text


def test_bottom_toolbar_drops_right_first_when_narrow() -> None:
    items = [("model", "m"), ("mcp", "3"), ("thread", "tid")]
    wide = fragment_list_to_text(_build_bottom_toolbar(items, cols=80))
    assert "thread:" in wide and "mcp:" in wide

    mid = fragment_list_to_text(_build_bottom_toolbar(items, cols=28))
    # thread 先丢
    assert "thread:" not in mid
    assert "model:" in mid

    tiny = fragment_list_to_text(_build_bottom_toolbar(items, cols=12))
    assert "model:" in tiny
    assert "thread:" not in tiny
