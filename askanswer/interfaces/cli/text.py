# 文本度量与格式化原语：终端可视宽度、CJK 感知截断、时间戳格式化。
#
# 都是无副作用的纯函数，只依赖标准库；被 render / progress / commands 广泛复用。
from __future__ import annotations

import os
import re
import time
import unicodedata

# 用于剥离字符串里的 ANSI 序列以便正确计算可视宽度
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    """去掉 ANSI 转义序列，剩下的才是真正显示出来的内容。"""
    return _ANSI_RE.sub("", s)


def _visual_width(s: str) -> int:
    """计算字符串在终端里的可视宽度，CJK 全角字符算 2 列。"""
    s = _strip_ansi(s)
    w = 0
    for ch in s:
        # East Asian Wide / Fullwidth 字符在等宽终端里占两列
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        else:
            w += 1
    return w


def _term_cols(min_cols: int = 20) -> int:
    """当前终端全列数（不封顶），失败时回退 80。

    用于输入框边框等需与终端等宽的绘制；欢迎面板等受控排版请用 ``_term_width``。
    """
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80
    return max(min_cols, cols)


def _term_width(max_width: int = 72) -> int:
    """当前终端宽度，限制在 [40, max_width] 区间，避免极端情况下排版崩溃。"""
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80
    return max(40, min(cols, max_width))


def _pad(content: str, inner_width: int) -> str:
    """把内容右侧补空格到指定可视宽度，配合边框面板使用。"""
    pad = inner_width - _visual_width(content)
    return content + " " * max(pad, 0)


def _truncate(s: str, limit: int = 60) -> str:
    """按可视宽度截断，保留 CJK 不被切半，结尾用省略号占一列。"""
    s = " ".join(str(s or "").split())
    if _visual_width(s) <= limit:
        return s
    out = ""
    for ch in s:
        # 给省略号留一列宽度
        if _visual_width(out + ch) > limit - 1:
            break
        out += ch
    return out + "…"


def _format_ts(ts: int) -> str:
    """把 epoch 秒格式化为 ``MM-DD HH:MM`` 紧凑形式（同一年）。"""
    if not ts:
        return "—"
    return time.strftime("%m-%d %H:%M", time.localtime(ts))
