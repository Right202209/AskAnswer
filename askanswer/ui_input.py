"""基于 prompt_toolkit 的输入会话。

提供：
- 命令历史回溯（↑/↓、Ctrl-R 反向搜索），文件持久化在 ~/.askanswer/history
- 斜杠命令补全：键入 ``/`` 自动弹出候选 + 一句话说明
- 真多行缓冲：Alt+Enter / Ctrl-J 换行，Enter 一次提交；可在缓冲区内上下编辑
- 软折行左侧保留 ``│`` gutter；上边框由 ``cli.repl`` 在 prompt 外打印
- 底部 status bar：紧凑三段（不右对齐铺满全宽，避免 resize reflow 残影）
- Ctrl-C 行为：第一次只清当前输入；2 秒内连按第二次才退出（与 Claude Code 一致）

设计取舍：
- 不再用 ``ui_select.is_interactive``：prompt_toolkit 自己会在非 TTY 时退化为
  ``input()``，省得我们再写一份 fallback；
- **不把全宽上边框放进 PTK message**：全宽行在缩窄窗口时被终端 reflow，
  PTK 的 erase 按旧高度上移会清不干净，导致整帧 prompt 重复堆叠
  （upstream #1933）。上/下边框改由 ``cli.repl`` print；
- bottom_toolbar 用 `` · `` 紧凑拼接、不空格右对齐，同样避免全宽 reflow；
- Application 上挂 reflow-aware ``_on_resize``：debounce 连发缩放、按列比放大
  erase、单帧重绘（禁止 CPR 二次 invalidate 叠字）；
- bottom_toolbar 是 callable —— 每次重绘自动取最新状态。
"""
from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import FormattedText, fragment_list_to_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

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
    ("/checkpoints", "列出当前会话快照",   "/checkpoints"),
    ("/undo",    "回退到上一个快照",       "/undo [n]"),
    ("/jump",    "跳转到指定快照",         "/jump <index>"),
    ("/fork",    "从快照分叉新会话",       "/fork [index]"),
    ("/audit",   "查看审计事件",           "/audit [thread] [--kind k] [--limit n]"),
    ("/usage",   "查看 token/工具用量",    "/usage [--days n] [--thread id]"),
    ("/export",  "导出会话",               "/export [thread] [--format md|json] [--out path]"),
    ("/import",  "导入 JSON 会话",         "/import <path.json>"),
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


# 命中即跳过写入 history 的常见敏感模式：API key、token、嵌入凭据的 DSN 等。
# 设计取舍：只做防御性过滤，不替代用户在敏感操作前手动 `export ASKANSWER_NO_HISTORY=1`。
_SECRET_HISTORY_RE = re.compile(
    r"""(?ix)
    sk-[A-Za-z0-9_-]{16,}              # OpenAI / 兼容 key
    | tvly-[A-Za-z0-9_-]{16,}          # Tavily key
    | xox[abprs]-[A-Za-z0-9-]{10,}     # Slack token
    | gh[pousr]_[A-Za-z0-9]{16,}       # GitHub token
    | AKIA[0-9A-Z]{12,}                # AWS access key id
    | (?:password|passwd|secret|token|api[_-]?key)\s*[:=]\s*\S+   # 显式赋值
    | [A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]+:[^@\s/]+@               # DSN(user:pass@host)
    """,
)


class _FilteredFileHistory(FileHistory):
    """跳过包含敏感模式的输入，避免 API key / DSN 被明文落盘到 history。"""

    def store_string(self, string: str) -> None:
        if not string:
            return
        if _SECRET_HISTORY_RE.search(string):
            return
        super().store_string(string)


# 整体配色：muted 边框 + brand 提示符 + 灰白底栏 + 高亮当前补全项
_PROMPT_STYLE = Style.from_dict({
    "prompt.border":      "#878787",
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

# 主提示 ``│ ❯ `` / 续行 ``│ · `` / 软折行 ``│   `` 的固定后缀（符号 + 空格）
_PROMPT_SUFFIX = "❯ "
_CONT_HARD_SUFFIX = "· "
_CONT_SOFT_SUFFIX = "  "


def _term_cols(min_cols: int = 20) -> int:
    """终端全列数（不封顶）。放在本模块避免依赖 ``cli`` 包造成循环导入。"""
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80
    return max(min_cols, cols)


def _prompt_message() -> FormattedText:
    """主提示符（不含全宽上边框，避免 resize reflow 残影）。"""
    return FormattedText([
        ("class:prompt.border", "│ "),
        ("class:prompt.gt", _PROMPT_SUFFIX),
    ])


def _prompt_continuation(width: int, line_number: int, wrap_count: int) -> FormattedText:
    """多行 / 软折行左侧 gutter，与主提示同宽（PTK 用 width 对齐换行）。

    ``wrap_count`` 是 PTK 传入的软折次数：0 = 硬换行后的新逻辑行，>0 = 软折行。
    不足 ``width`` 时右侧补空格，避免 gutter 比主提示窄导致 wrap/光标错位。
    """
    del line_number
    is_soft_wrap = wrap_count > 0
    suffix = _CONT_SOFT_SUFFIX if is_soft_wrap else _CONT_HARD_SUFFIX
    style = "class:prompt.border" if is_soft_wrap else "class:prompt.gt"
    fragments: list[tuple[str, str]] = [
        ("class:prompt.border", "│ "),
        (style, suffix),
    ]
    cur = get_cwidth(fragment_list_to_text(fragments))
    target = max(int(width or 0), cur)
    if cur < target:
        fragments.append(("class:prompt.border", " " * (target - cur)))
    return FormattedText(fragments)


def _build_bottom_toolbar(
    items: list[tuple[str, str]],
    cols: int,
) -> list[tuple[str, str]]:
    """紧凑底栏：`` ● model:… · mcp:… · thread:…``，不空格铺满全宽。

    宽度不够时按右→中丢弃（thread 先、mcp 次），避免右对齐 padding 触发
    终端 reflow（prompt_toolkit #1933）。
    """
    items_map = dict(items)
    segments: list[tuple[str, str]] = [
        ("model", str(items_map.get("model") or "—")),
    ]
    mcp_val = items_map.get("mcp")
    if mcp_val:
        segments.append(("mcp", f"{mcp_val} tools"))
    segments.append(("thread", str(items_map.get("thread") or "?")))

    def segs_width(segs: list[tuple[str, str]]) -> int:
        # " ● " + sum(len(label)+1+len(value)) + " · " between
        w = 3  # prefix
        for i, (label, value) in enumerate(segs):
            if i:
                w += 3  # " · "
            w += len(label) + 1 + len(value)
        return w

    # 至少保留 model；其余从右往左丢
    while len(segments) > 1 and segs_width(segments) > cols:
        segments.pop()

    parts: list[tuple[str, str]] = [("class:bottom-toolbar.dot", " ● ")]
    for i, (label, value) in enumerate(segments):
        if i:
            parts.append(("class:bottom-toolbar", " · "))
        parts.append(("class:bottom-toolbar.lbl", f"{label}:"))
        parts.append(("class:bottom-toolbar.val", value))
    return parts


# SIGWINCH 在拖拽缩放时会连发；合并成一次 erase+repaint，避免叠帧（同一行画多次）。
_RESIZE_DEBOUNCE_S = 0.05


def _reflow_erase_up_rows(
    cursor_y: int,
    *,
    last_columns: int | None,
    new_columns: int | None,
    new_rows: int | None,
) -> int:
    """计算 resize 后 erase 前应 ``cursor_up`` 的行数。

    缩窄时终端会把原先「一行」折成多行，PTK 记录的 ``_cursor_pos.y`` 仍是
    旧布局下的相对行号，必须按列比放大，否则 erase 起点落在旧内容中间，
    新帧叠上去后同一行文字会残留并被再画一次（upstream #1933）。

    注意：不要用 ``last_screen.height`` 作下限——PTK 常把 screen 拉满
    ``min_available_height``（几乎整屏），会误删历史输出。
    """
    up = max(int(cursor_y), 0)
    if new_columns is not None and new_columns > 0:
        if last_columns is not None and last_columns > new_columns:
            factor = max(1, (last_columns + new_columns - 1) // new_columns)
            # 光标行本身也会 reflow：用 (y+1)*factor 估到 reflow 后 UI 顶部
            # +2：底栏 / 外置边框 reflow 余量
            up = (up + 1) * factor + 2
        elif last_columns is None:
            # 列宽未知时略多擦几行，防连续 resize 丢了 last_columns
            up = up + 3
        # 变宽：终端已把多行折回，up=cursor_y 足够（略多无害）
    if new_rows is not None and new_rows > 0:
        up = min(up, max(new_rows - 1, cursor_y))
    return max(up, 0)


def _install_resize_safe_erase(app) -> None:
    """替换 Application._on_resize：合并连发缩放，单次 erase 后只重绘一帧。

    终端把全宽行折成多行后，PTK 记录的 ``_cursor_pos.y`` 会小于真实占高，
    默认 erase 上移不够 → 旧帧残留、新帧再画一遍，表现为同一行文字重复
    （upstream #1933）。此处：
    1. debounce 合并 SIGWINCH + size-poll 连发；
    2. reflow-aware erase（清掉残留行）；
    3. ``layout.reset()`` 清掉 Window 的 scroll 偏移；
    4. 直接设定 ``_min_available_height``，**只重绘一次**——禁止 CPR 后再
       invalidate 第二帧（第二帧在物理屏与 ``_last_screen`` 不一致时会叠字）。
    """
    if getattr(app, "_askanswer_resize_patched", False):
        return
    app._askanswer_resize_patched = True  # type: ignore[attr-defined]
    renderer = app.renderer

    def _apply_resize() -> None:
        app._askanswer_resize_timer = None  # type: ignore[attr-defined]
        if not getattr(app, "is_running", False) or getattr(app, "is_done", True):
            return
        if getattr(app, "_askanswer_resize_busy", False):
            # 重入：标一下，当前帧结束后再跑一轮
            app._askanswer_resize_again = True  # type: ignore[attr-defined]
            return

        app._askanswer_resize_busy = True  # type: ignore[attr-defined]
        try:
            while True:
                app._askanswer_resize_again = False  # type: ignore[attr-defined]
                _erase_and_repaint()
                if not getattr(app, "_askanswer_resize_again", False):
                    break
        finally:
            app._askanswer_resize_busy = False  # type: ignore[attr-defined]

    def _erase_and_repaint() -> None:
        output = app.output
        cp = renderer._cursor_pos
        last_size = getattr(renderer, "_last_size", None)
        # 连续 SIGWINCH 时上一次 reset 会清空 _last_size；用我们自己记的列数兜底
        last_columns = getattr(app, "_askanswer_last_columns", None)
        if last_size is not None:
            last_columns = last_size.columns
        try:
            new_size = output.get_size()
        except Exception:
            new_size = None

        if cp.x > 0:
            output.cursor_backward(cp.x)

        up = _reflow_erase_up_rows(
            cp.y,
            last_columns=last_columns,
            new_columns=new_size.columns if new_size is not None else None,
            new_rows=new_size.rows if new_size is not None else None,
        )
        if up > 0:
            output.cursor_up(up)
        output.erase_down()
        output.reset_attributes()
        # 保持 autowrap 关闭直到 _redraw：中途 enable 会让残留半行被终端再折一次
        try:
            output.disable_autowrap()
        except Exception:
            pass
        output.flush()

        renderer.reset(leave_alternate_screen=False)
        # 重置滚动偏移，迫使 BufferControl 按新列宽重算 wrap → 光标映射
        try:
            app.layout.reset()
        except Exception:
            pass

        if new_size is not None:
            app._askanswer_last_columns = new_size.columns  # type: ignore[attr-defined]
            # 光标已在擦除后的 UI 顶部：先占满下方行数，单帧即可画出底栏，
            # 不必等 CPR 再 invalidate（那会变成第二帧叠画）。
            renderer._min_available_height = max(new_size.rows, 1)

        # CPR 仍请求，用于后续键入时校正；但不再为此安排二次重绘
        app._request_absolute_cursor_position()
        app._redraw()

    def _on_resize() -> None:
        loop = getattr(app, "loop", None)
        if loop is None:
            _apply_resize()
            return
        timer = getattr(app, "_askanswer_resize_timer", None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass
        try:
            app._askanswer_resize_timer = loop.call_later(  # type: ignore[attr-defined]
                _RESIZE_DEBOUNCE_S, _apply_resize
            )
        except Exception:
            _apply_resize()

    app._on_resize = _on_resize  # type: ignore[method-assign]


def make_session(get_status: Callable[[], list[tuple[str, str]]]) -> PromptSession:
    """构造一个配置好的 PromptSession。

    ``get_status`` 是个无参 callable，每次重绘 toolbar 时调用，返回 ``[(label, value), …]``。
    用 callable 而不是固定字符串：模型/线程切换后 toolbar 会自动反映新状态。
    """

    def bottom_toolbar():
        return _build_bottom_toolbar(get_status(), _term_cols())

    bindings = KeyBindings()

    # multiline=True 时默认 Enter 插入换行；我们反转：Enter 提交，Alt+Enter / Ctrl-J 换行
    @bindings.add("enter")
    def _submit(event) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")  # Alt+Enter（多数终端）
    def _newline_alt(event) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-j")
    def _newline_ctrl_j(event) -> None:
        event.current_buffer.insert_text("\n")

    return PromptSession(
        history=_FilteredFileHistory(str(_history_path())),
        completer=_SlashCompleter(),
        complete_while_typing=True,
        bottom_toolbar=bottom_toolbar,
        key_bindings=bindings,
        style=_PROMPT_STYLE,
        editing_mode=EditingMode.EMACS,
        mouse_support=False,
        multiline=True,
        wrap_lines=True,
        prompt_continuation=_prompt_continuation,
    )


def read_line(session: PromptSession) -> str | None:
    """读一段用户输入（可多行）。

    返回值：
        - ``None``      → Ctrl-D 或 2 秒内二连 Ctrl-C，调用方应退出
        - ``""``        → 单次 Ctrl-C，调用方应继续下一轮（保持原 thread）
        - 其它字符串    → 用户输入；硬换行保留为 ``\\n``
    """
    global _last_interrupt_ts

    def _pre_run() -> None:
        try:
            app = get_app()
            _install_resize_safe_erase(app)
            # 首帧记下列宽，供连续 SIGWINCH 时 last_size 已被 reset 的情况兜底
            try:
                app._askanswer_last_columns = app.output.get_size().columns  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

    try:
        text = session.prompt(_prompt_message, pre_run=_pre_run)
    except EOFError:
        return None
    except KeyboardInterrupt:
        now = time.monotonic()
        if now - _last_interrupt_ts < _DOUBLE_CTRLC_WINDOW:
            return None  # 二连 → 退出
        _last_interrupt_ts = now
        return ""  # 单次 → 调用方提示并继续

    # 兼容旧习惯：行尾单独的 ``\`` 续行符规范成换行后去掉
    if "\\\n" in text or text.endswith("\\"):
        text = text.replace("\\\n", "\n")
        if text.endswith("\\"):
            text = text[:-1]
    return text


__all__ = [
    "SLASH_COMMANDS",
    "cmd_meta",
    "make_session",
    "read_line",
]
