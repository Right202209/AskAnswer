"""基于 prompt_toolkit 的输入会话。

提供：
- 命令历史回溯（↑/↓、Ctrl-R 反向搜索），文件持久化在 ~/.askanswer/history
- 斜杠命令补全：键入 ``/`` 自动弹出候选 + 一句话说明
- 反斜杠续行：行尾 ``\\`` 表示“还有下一行”，多行被拼成一段提交
- 底部 status bar：实时展示当前 thread / model / mcp 连接数
- Ctrl-C 行为：第一次只清当前输入；2 秒内连按第二次才退出（与 Claude Code 一致）

设计取舍：
- 不再用 ``ui_select.is_interactive``：prompt_toolkit 自己会在非 TTY 时退化为
  ``input()``，省得我们再写一份 fallback；
- 顶/底边框仍然由 ``cli`` 自己 print（保留视觉），prompt_toolkit 只负责中间一行；
- bottom_toolbar 是 callable —— 每次重绘自动取最新状态，无需我们手动刷新。
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style


# 斜杠命令清单 + 一句话说明；同时供 ``handle_command`` 之外的 ``/help <cmd>`` 使用。
# 顺序也是 ``/help`` 列表展示顺序。
SLASH_COMMANDS: list[tuple[str, str, str]] = [
    ("/help",    "显示帮助",              "/help [cmd]"),
    ("/clear",   "清屏并开始新会话",       "/clear"),
    ("/status",  "查看当前会话信息",       "/status"),
    ("/model",   "查看或切换模型",         "/model [<provider:name>]"),
    ("/mcp",     "管理 MCP 服务",          "/mcp [list|tools|remove|<url>]"),
    ("/threads", "列出历史会话",           "/threads [关键词]"),
    ("/resume",  "恢复指定会话",           "/resume <序号|id>"),
    ("/title",   "给当前会话命名",         "/title <名字>"),
    ("/delete",  "删除会话",               "/delete <序号|id>"),
    ("/exit",    "退出 (/quit, /q, Ctrl-D)", "/exit"),
]


# 把 SLASH_COMMANDS 转成 dict 方便 ``cmd_meta`` 查询
_CMD_META: dict[str, tuple[str, str]] = {
    name: (desc, usage) for name, desc, usage in SLASH_COMMANDS
}


def cmd_meta(name: str) -> tuple[str, str] | None:
    """根据 ``/foo`` 返回 (description, usage)；找不到时返回 None。"""
    return _CMD_META.get(name)


class _SlashCompleter(Completer):
    """只在用户键入以 ``/`` 开头的内容时弹出候选。"""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        # 简单前缀匹配；prompt_toolkit 会用 display + display_meta 渲染候选项
        for name, desc, _ in SLASH_COMMANDS:
            if name.startswith(text):
                yield Completion(
                    name,
                    start_position=-len(text),
                    display=name,
                    display_meta=desc,
                )


def _history_path() -> Path:
    """复用 ``persistence.default_db_path`` 的目录约定，把 history 放在同级。"""
    explicit = os.environ.get("ASKANSWER_DB_PATH")
    if explicit:
        base = Path(explicit).expanduser().parent
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg).expanduser() / "askanswer" if xdg else Path.home() / ".askanswer"
    base.mkdir(parents=True, exist_ok=True)
    return base / "history"


# 整体配色：橙色提示符 + 灰白底栏 + 高亮当前补全项
_PROMPT_STYLE = Style.from_dict({
    "prompt.border":      "#ffaf00",
    "prompt.gt":          "#ffaf00 bold",
    "bottom-toolbar":     "#878787 bg:default",
    "bottom-toolbar.dot": "#5fd787",
    "bottom-toolbar.lbl": "#878787",
    "bottom-toolbar.val": "#87afff",
    # 补全菜单：当前项橙底黑字，其它项暗灰
    "completion-menu.completion":         "bg:#3a3a3a #d7d7d7",
    "completion-menu.completion.current": "bg:#ffaf00 #000000",
    "completion-menu.meta.completion":    "bg:#3a3a3a #878787",
})


# 第一次按 Ctrl-C 的时间戳；超过 ``_DOUBLE_CTRLC_WINDOW`` 秒第二次才生效退出。
_last_interrupt_ts = 0.0
_DOUBLE_CTRLC_WINDOW = 2.0


def make_session(get_status: Callable[[], list[tuple[str, str]]]) -> PromptSession:
    """构造一个配置好的 PromptSession。

    ``get_status`` 是个无参 callable，每次重绘 toolbar 时调用，返回 ``[(label, value), …]``。
    用 callable 而不是固定字符串：模型/线程切换后 toolbar 会自动反映新状态。
    """

    def bottom_toolbar():
        items = get_status()
        parts: list[tuple[str, str]] = [("class:bottom-toolbar.dot", " ● ")]
        for i, (label, value) in enumerate(items):
            if i > 0:
                parts.append(("class:bottom-toolbar", "  ·  "))
            parts.append(("class:bottom-toolbar.lbl", f"{label}: "))
            parts.append(("class:bottom-toolbar.val", str(value)))
        return parts

    bindings = KeyBindings()
    # 留个钩子：以后想加 “/” 即开补全菜单 / Esc-Enter 多行 / Ctrl-L 清屏 都接在这。

    return PromptSession(
        history=FileHistory(str(_history_path())),
        completer=_SlashCompleter(),
        complete_while_typing=True,
        bottom_toolbar=bottom_toolbar,
        key_bindings=bindings,
        style=_PROMPT_STYLE,
        editing_mode=EditingMode.EMACS,
        mouse_support=False,
    )


def read_line(
    session: PromptSession,
    *,
    continuation_indent: int = 2,
) -> str | None:
    """读一行（或反斜杠续行的多行）。

    返回值：
        - ``None``      → Ctrl-D 或 2 秒内二连 Ctrl-C，调用方应退出
        - ``""``        → 单次 Ctrl-C，调用方应继续下一轮（保持原 thread）
        - 其它字符串    → 用户输入；多行 ``\\`` 续行已被合并为 ``\\n``
    """
    global _last_interrupt_ts

    # 主输入行的提示符：``│ > ``，用 FormattedText 让边框/箭头单独着色
    main_prompt = FormattedText([
        ("class:prompt.border", "│ "),
        ("class:prompt.gt",     "> "),
    ])
    cont_prompt = FormattedText([
        ("class:prompt.border", "│ "),
        ("class:prompt.gt",     "·" + " " * continuation_indent),
    ])

    parts: list[str] = []
    prompt_text = main_prompt
    while True:
        try:
            line = session.prompt(prompt_text)
        except EOFError:
            return None
        except KeyboardInterrupt:
            now = time.monotonic()
            if now - _last_interrupt_ts < _DOUBLE_CTRLC_WINDOW:
                return None  # 二连 → 退出
            _last_interrupt_ts = now
            return ""        # 单次 → 调用方提示并继续

        # 行尾反斜杠：吞掉它，进入续行
        if line.endswith("\\"):
            parts.append(line[:-1])
            prompt_text = cont_prompt
            continue
        parts.append(line)
        return "\n".join(parts)


__all__ = [
    "SLASH_COMMANDS",
    "cmd_meta",
    "make_session",
    "read_line",
]
