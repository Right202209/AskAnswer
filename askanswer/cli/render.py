# 终端渲染：欢迎/状态面板、帮助清单、答案与错误的 Markdown 输出。
#
# 纯展示层，无副作用地读取 model / mcp / persistence 状态并画出来；不参与图执行。
from __future__ import annotations

import os
from pathlib import Path

from rich import box
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..load import current_model_label, model
from ..mcp import get_manager as _mcp_manager
from ..persistence import get_persistence
from ..ui_input import SLASH_COMMANDS, cmd_meta
from .theme import C, _console


def _current_model_name() -> str:
    """尽力拿到当前模型可读名（先用 load.current_model_label，再回退到对象属性）。"""
    label = current_model_label()
    if label:
        return label
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def _welcome_info(thread_id: str | None) -> str:
    """右下角小字：`model · mcp:N · thread:abcd123`。"""
    parts = [_current_model_name()]
    try:
        n = len(_mcp_manager().list_servers())
    except Exception:
        n = 0
    parts.append(f"mcp:{n}")
    if thread_id:
        parts.append(f"thread:{thread_id[:7]}")
    return " · ".join(parts)


def welcome_box(thread_id: str | None = None) -> None:
    """启动时画的欢迎面板。

    终端宽度 < 80 列：降级到纯文字（单列），避免折行/裁剪难看。
    >= 80 列：圆角面板，右下角放 model · mcp · thread 一行小字。
    """
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80
    info_line = _welcome_info(thread_id)

    # 窄终端：直接打几行字，不上面板
    if cols < 80:
        _console.print()
        _console.print("[brand]✻[/] [accent]AskAnswer[/]")
        _console.print("[muted]AI 助手 · 输入 /help 查看命令, /exit 退出[/]")
        _console.print(f"[muted]cwd: {Path.cwd()}[/]")
        if info_line:
            _console.print(f"[muted]{info_line}[/]")
        _console.print()
        return

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(no_wrap=True, justify="right")

    grid.add_row(
        Text.from_markup("[brand]✻[/] [accent]AskAnswer[/]"),
        "",
    )
    grid.add_row(
        Text.from_markup("[muted]AI assistant · LangGraph powered[/]"),
        "",
    )
    grid.add_row("", "")
    grid.add_row(
        Text.from_markup(
            "[muted]输入[/] [info]/help[/] [muted]查看命令, [/]"
            "[info]/exit[/] [muted]退出[/]"
        ),
        "",
    )
    grid.add_row("", "")
    grid.add_row(
        Text.from_markup(f"[muted]cwd: {Path.cwd()}[/]"),
        Text.from_markup(f"[muted]{info_line}[/]") if info_line else "",
    )

    _console.print()
    _console.print(
        Panel(
            grid,
            border_style="brand",
            box=box.ROUNDED,
            padding=(0, 2),
            width=min(cols - 2, 100),
        )
    )


def tips_block() -> None:
    """欢迎面板下方的“使用小贴士”。"""
    _console.print()
    _console.print(" [bold]Tips for getting started:[/]")
    _console.print()
    _console.print(" [subtle]1.[/] 提出任何问题，我会进行搜索并整理答案")
    _console.print(" [subtle]2.[/] 问题越具体，结果越精准")
    _console.print(" [subtle]3.[/] 输入 [info]/help[/] 查看所有命令")
    _console.print()


def help_block(target: str | None = None) -> None:
    """``/help`` 输出命令清单；``/help <cmd>`` 输出单条命令的详细用法。"""
    if target:
        # 容错：``/help help`` 与 ``/help /help`` 都能命中
        key = target if target.startswith("/") else f"/{target}"
        meta = cmd_meta(key)
        print()
        if meta is None:
            print(f"  {C.RED}未知命令：{C.RESET}{key}  "
                  f"({C.CYAN}/help{C.RESET} 看全部)")
            print()
            return
        desc, usage = meta
        print(f"  {C.BOLD}{key}{C.RESET}  {C.DIM}{desc}{C.RESET}")
        print(f"   {C.DIM}用法：{C.RESET}{C.CYAN}{usage}{C.RESET}")
        # 几条特例，给一个具体范例 —— 减少“看不懂格式”的回头问
        examples = _help_examples(key)
        if examples:
            print(f"   {C.DIM}示例：{C.RESET}")
            for ex in examples:
                print(f"     {C.DIM}·{C.RESET} {C.CYAN}{ex}{C.RESET}")
        print()
        return

    # 全清单：直接读 SLASH_COMMANDS（与补全菜单单一真源）
    print()
    print(f" {C.BOLD}Commands{C.RESET}  "
          f"{C.DIM}(/<Tab> 自动补全 · /help <cmd> 查看详细){C.RESET}")
    for name, desc, _usage in SLASH_COMMANDS:
        print(f"   {C.CYAN}{name:<9}{C.RESET} {desc}")
    print(f"   {C.CYAN}!<cmd>{C.RESET}    直接执行 shell 命令 (如 !ls -la)")
    print()
    print(f" {C.DIM}快捷键：↑/↓ 历史 · Ctrl-R 反向搜索 · "
          f"行尾 \\ 多行续行 · Ctrl-C 取消（连按二次退出）{C.RESET}")
    print()


def _help_examples(cmd: str) -> list[str]:
    """单条命令的 hands-on 示例，方便用户照抄。"""
    table = {
        "/model":   ["/model gpt-4o-mini", "/model anthropic:claude-3-5-sonnet"],
        "/mcp":     ["/mcp https://example.com/mcp my-server",
                     "/mcp add_stdio fs npx -y @modelcontextprotocol/server-filesystem /tmp",
                     "/mcp health", "/mcp tools my-server", "/mcp remove my-server"],
        "/threads": ["/threads", "/threads sql 关键词"],
        "/resume":  ["/resume 1", "/resume 1e14b9b"],
        "/title":   ["/title 周三的 SQL 调试"],
        "/delete":  ["/delete 1", "/delete 1e14b9b"],
        "/checkpoints": ["/checkpoints"],
        "/undo":    ["/undo", "/undo 2", "/undo 2 --label before-refactor", "/undo --label before-refactor"],
        "/jump":    ["/jump 3"],
        "/fork":    ["/fork", "/fork 2"],
        "/audit":   ["/audit", "/audit 1 --limit 20", "/audit --kind tool_call"],
        "/usage":   ["/usage --days 1", "/usage --thread 1"],
        "/export":  ["/export 1 --format md --out /tmp/thread.md",
                     "/export current --format json --out /tmp/thread.json"],
        "/import":  ["/import /tmp/thread.json"],
    }
    return table.get(cmd, [])


def status_block(thread_id: str) -> None:
    """/status 输出：当前线程 ID、CWD、模型、MCP 连接状态、持久化信息。"""
    from .commands._common import _current_tenant

    # 一对 (label, value) 行；None 值会被跳过
    rows: list[tuple[str, str]] = [("thread", thread_id)]
    try:
        meta = get_persistence().get_meta(thread_id)
    except Exception:
        meta = None
    if meta and meta.title:
        rows.append(("title", meta.title))
    rows.append(("cwd", str(Path.cwd())))
    rows.append(("model", current_model_label()))
    try:
        pm = get_persistence()
        thread_count = len(pm.list_threads(limit=10000, tenant_id=_current_tenant()))
        rows.append(("store", f"{pm.db_path}  [subtle]({thread_count} threads)[/]"))
    except Exception:
        pass
    servers = _mcp_manager().list_servers()
    if servers:
        summary = ", ".join(f"{s['name']}({s['tools']})" for s in servers)
        rows.append(("mcp", summary))
    else:
        rows.append(("mcp", "[subtle]（未连接，/mcp <url> 添加）[/]"))

    # 用 Table 做左右两列的等宽对齐，免去手算 padding。
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="subtle", justify="right", no_wrap=True)
    grid.add_column()
    for label, value in rows:
        grid.add_row(f"{label}:", value)

    _console.print()
    _console.print(
        Panel(
            grid,
            title="[bold]Status[/]",
            title_align="left",
            border_style="muted",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        )
    )
    _console.print()


def render_answer(answer: str) -> None:
    """把最终答案以 Markdown 渲染输出。"""
    _console.print()
    _console.print(Rule(title="[subtle]Answer[/]", style="muted", align="left"))
    _console.print(Padding(Markdown(answer or "_(空答案)_"), (0, 2)))
    _console.print()


def render_error(message: str) -> None:
    """统一的错误样式：红色叉号 + 灰色细节。"""
    _console.print()
    _console.print("  [danger]✗ 运行失败[/]")
    _console.print(f"  [subtle]{message}[/]")
    _console.print()
