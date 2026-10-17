# CLI 视觉主题：rich 控制台与 ANSI 颜色常量的单一真源。
#
# ``_console`` 承载所有 Panel / Table / Markdown 渲染；``Theme`` 是调色板的唯一
# 入口（改配色只动这里）。``class C`` 是给 print f-string 路径保留的兼容层。
from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

# rich 控制台：所有 Panel / Table / Markdown 都走它。``Theme`` 是 cli 视觉的
# 单一真源 —— 调色板调整只动这里一处。``class C`` 是给 print f-string 路径
# 保留的兼容层，原有调用不强制迁移。
_THEME = Theme({
    "brand": "color(214)",      # 主品牌橙
    "accent": "color(214) bold",
    "info": "color(117)",       # 命令 / ID / 高亮值
    "success": "color(114)",    # 成功提示
    "warning": "color(178)",    # 警告 / 待确认
    "danger": "color(203)",     # 错误 / 高风险
    "muted": "color(240)",      # 边框 / 弱化
    "subtle": "dim",            # 相对调暗
})
_console = Console(theme=_THEME, highlight=False)


class C:
    """ANSI 颜色与样式常量集合，方便在 print 时拼接。"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ORANGE = "\033[38;5;214m"
    GOLD = "\033[38;5;178m"
    GRAY = "\033[38;5;240m"
    CYAN = "\033[38;5;117m"
    GREEN = "\033[38;5;114m"
    RED = "\033[38;5;203m"
