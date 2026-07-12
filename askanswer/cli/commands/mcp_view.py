# ``/mcp list`` / ``/mcp tools`` 的表格渲染（从 mcp.py 拆出的纯展示层）。
from __future__ import annotations

from rich import box
from rich.table import Table

from ...mcp import get_manager as _mcp_manager
from ..text import _format_ts
from ..theme import _console


def _print_mcp_servers(*, verbose: bool = False) -> None:
    """打印已连接的 MCP 服务清单。verbose 模式下附带健康状态与最近探测时间。"""
    servers = _mcp_manager().list_servers()
    _console.print()
    if not servers:
        _console.print("  [subtle]（暂未连接 MCP 服务）[/]")
        _console.print()
        return
    _console.print(f"  [bold]MCP Servers ({len(servers)})[/]")
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="subtle",
        border_style="muted",
        padding=(0, 1),
        expand=False,
    )
    table.add_column("", width=1, no_wrap=True)
    table.add_column("name", style="info", no_wrap=True)
    table.add_column("transport", style="subtle", no_wrap=True)
    table.add_column("tools", justify="right", style="subtle", no_wrap=True)
    if verbose:
        table.add_column("status", no_wrap=True)
        table.add_column("checked", style="subtle", no_wrap=True)
    table.add_column("url", style="subtle", no_wrap=True, overflow="ellipsis", max_width=46)
    for s in servers:
        connected = s.get("status", "connected") == "connected"
        dot = "[success]●[/]" if connected else "[danger]○[/]"
        row = [dot, s["name"], s["transport"], str(s["tools"])]
        if verbose:
            status_cell = "[success]connected[/]" if connected else "[danger]disconnected[/]"
            row.extend([status_cell, _format_ts(s.get("last_checked") or 0)])
        row.append(s.get("url") or "")
        table.add_row(*row)
    _console.print(table)
    _console.print()


def _print_mcp_tools(server: str | None) -> None:
    """打印某个 server（或全部 server）下的工具列表。"""
    tools = _mcp_manager().list_tools(server=server)
    _console.print()
    if not tools:
        hint = f"{server} 无工具 / 未连接" if server else "暂无 MCP 工具，使用 /mcp <url> 添加"
        _console.print(f"  [subtle]（{hint}）[/]")
        _console.print()
        return
    scope = f" ({server})" if server else ""
    _console.print(f"  [bold]MCP Tools{scope} · {len(tools)}[/]")
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="subtle",
        border_style="muted",
        padding=(0, 1),
        expand=False,
    )
    table.add_column("name", style="info", no_wrap=True)
    table.add_column("description", style="subtle", no_wrap=True, overflow="ellipsis", max_width=60)
    for t in tools:
        table.add_row(t["name"], t.get("description") or "")
    _console.print(table)
    _console.print()
