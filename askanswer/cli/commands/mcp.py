# ``/mcp`` 子命令：连接（URL / stdio）、健康探测、列表、工具、移除，并同步 profile。
from __future__ import annotations

from rich import box
from rich.table import Table

from ... import mcp_profile
from ...audit import log_event
from ...mcp import get_manager as _mcp_manager
from ...registry import get_registry
from ..render import render_error
from ..text import _format_ts, _truncate
from ..theme import C, _console
from ._common import _split_args
from .mcp_view import _print_mcp_servers, _print_mcp_tools


def mcp_help_block() -> None:
    """/mcp 子命令帮助。"""
    print()
    print(f" {C.BOLD}/mcp{C.RESET}")
    print(f"   {C.CYAN}/mcp <url> [name]{C.RESET}      连接一个 MCP 服务 (HTTP/SSE)")
    print(f"   {C.CYAN}/mcp add_stdio <name> <cmd> [args…]{C.RESET}  以子进程方式启动 stdio 服务")
    print(f"   {C.CYAN}/mcp list [-v]{C.RESET}          列出已连接的 MCP 服务 (-v 显示健康详情)")
    print(f"   {C.CYAN}/mcp health [name]{C.RESET}      探测服务健康状态并刷新工具")
    print(f"   {C.CYAN}/mcp tools [server]{C.RESET}    列出工具 (可选按 server 过滤)")
    print(f"   {C.CYAN}/mcp remove <name>{C.RESET}     断开指定服务")
    print()


def handle_mcp_command(args: str, *, thread_id: str) -> None:
    """/mcp 子命令分发：list / tools / remove / 添加（URL 直接形式）。"""
    if not args:
        # 无参：先列出已连服务，再展示 /mcp 的帮助
        _print_mcp_servers()
        mcp_help_block()
        return

    first, _, rest = args.partition(" ")
    rest = rest.strip()
    first_lc = first.lower()

    if first_lc in {"list", "ls"}:
        _print_mcp_servers(verbose=rest.strip() in {"-v", "--verbose"})
    elif first_lc in {"health", "ping"}:
        _mcp_health_command(rest or None, thread_id=thread_id)
    elif first_lc in {"add_stdio", "stdio"}:
        _add_mcp_stdio(rest, thread_id=thread_id)
    elif first_lc in {"tools", "tool"}:
        _print_mcp_tools(rest or None)
    elif first_lc in {"remove", "rm", "disconnect"}:
        if not rest:
            print(f"\n  {C.RED}用法：{C.RESET}/mcp remove <name>\n")
            return
        _remove_mcp_server(rest)
    elif first_lc in {"help", "-h", "--help"}:
        mcp_help_block()
    elif first.startswith(("http://", "https://")):
        # 直接传 URL 等价于添加一个新服务
        _add_mcp_url(first, rest or None, thread_id=thread_id)
    else:
        print(
            f"\n  {C.RED}无法识别 /mcp 参数：{C.RESET}{args}\n"
            f"  {C.DIM}提示：URL 需以 http:// 或 https:// 开头{C.RESET}"
        )
        mcp_help_block()


def _add_mcp_url(url: str, name: str | None, *, thread_id: str) -> None:
    """连接一个 HTTP/SSE 类的 MCP 服务，并刷新工具注册表。"""
    try:
        registered = _mcp_manager().add_url(url, name=name)
    except Exception as exc:
        log_event(
            kind="mcp_connect",
            thread_id=thread_id,
            args_summary=url,
            error=str(exc),
            immediate=True,
        )
        render_error(f"MCP 连接失败: {exc}")
        return
    # 注册表里 mcp:* 这一片需要重新拉取
    get_registry().refresh_mcp()
    # 连接成功后写入 profile，下次启动自动重连；失败仅告警不阻塞。
    # transport 与 manager 的猜测口径一致（/sse → sse，否则 streamable_http），
    # 保证 profile 回放时不会把 SSE 端点错当成 streamable_http。
    transport = "sse" if url.rstrip("/").lower().endswith("/sse") else "streamable_http"
    _save_mcp_profile_entry(
        {"name": registered, "transport": transport, "url": url}
    )
    tools = _mcp_manager().list_tools(server=registered)
    log_event(
        kind="mcp_connect",
        thread_id=thread_id,
        tool_name=registered,
        args_summary=url,
        result_size=len(tools),
        immediate=True,
    )
    print()
    print(
        f"  {C.GREEN}✓ 已连接 MCP:{C.RESET} {C.BOLD}{registered}{C.RESET}  "
        f"{C.DIM}{url}{C.RESET}"
    )
    if tools:
        print(f"  {C.DIM}工具 ({len(tools)}):{C.RESET}")
        for t in tools:
            desc = _truncate(t.get("description") or "", 56)
            print(f"    {C.CYAN}{t['name']}{C.RESET}  {C.DIM}{desc}{C.RESET}")
    else:
        print(f"  {C.DIM}（未发现工具）{C.RESET}")
    print()


def _add_mcp_stdio(args: str, *, thread_id: str) -> None:
    """``/mcp add_stdio <name> <command> [args…]``：启动一个 stdio 子进程 server。"""
    parts = _split_args(args)
    if parts is None:
        return
    if len(parts) < 2:
        print(
            f"\n  {C.RED}用法：{C.RESET}/mcp add_stdio <name> <command> [args…]\n"
            f"  {C.DIM}例：/mcp add_stdio fs npx -y @modelcontextprotocol/server-filesystem /tmp{C.RESET}\n"
        )
        return
    name, command, cmd_args = parts[0], parts[1], parts[2:]
    try:
        registered = _mcp_manager().add_stdio(name=name, command=command, args=cmd_args)
    except Exception as exc:
        log_event(
            kind="mcp_connect",
            thread_id=thread_id,
            args_summary=f"stdio:{command}",
            error=str(exc),
            immediate=True,
        )
        render_error(f"MCP 连接失败: {exc}")
        return
    get_registry().refresh_mcp()
    _save_mcp_profile_entry(
        {
            "name": registered,
            "transport": "stdio",
            "command": command,
            "args": cmd_args,
        }
    )
    tools = _mcp_manager().list_tools(server=registered)
    log_event(
        kind="mcp_connect",
        thread_id=thread_id,
        tool_name=registered,
        args_summary=f"stdio:{command}",
        result_size=len(tools),
        immediate=True,
    )
    print()
    print(
        f"  {C.GREEN}✓ 已连接 MCP:{C.RESET} {C.BOLD}{registered}{C.RESET}  "
        f"{C.DIM}stdio · {command}{C.RESET}"
    )
    print(f"  {C.DIM}工具 ({len(tools)}){C.RESET}" if tools else f"  {C.DIM}（未发现工具）{C.RESET}")
    print()


def _mcp_health_command(name: str | None, *, thread_id: str) -> None:
    """``/mcp health [name]``：探测健康状态、刷新注册表、渲染结果表格。"""
    try:
        results = _mcp_manager().health_check(name)
    except Exception as exc:
        render_error(f"健康探测失败: {exc}")
        return
    # 探测后有 server 可能翻红/转绿，工具集需要同步；registry 会跳过 disconnected。
    get_registry().refresh_mcp()
    _console.print()
    if not results:
        hint = f"未找到 MCP 服务：{name}" if name else "暂未连接 MCP 服务"
        _console.print(f"  [subtle]（{hint}）[/]")
        _console.print()
        return
    _console.print(f"  [bold]MCP Health ({len(results)})[/]")
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
    table.add_column("status", no_wrap=True)
    table.add_column("tools", justify="right", style="subtle", no_wrap=True)
    table.add_column("checked", style="subtle", no_wrap=True)
    table.add_column("error", style="subtle", no_wrap=True, overflow="ellipsis", max_width=32)
    for s in results:
        connected = s.get("status") == "connected"
        dot = "[success]●[/]" if connected else "[danger]○[/]"
        status_cell = "[success]connected[/]" if connected else "[danger]disconnected[/]"
        table.add_row(
            dot,
            s["name"],
            s["transport"],
            status_cell,
            str(s["tools"]),
            _format_ts(s.get("last_checked") or 0),
            s.get("last_error") or "",
        )
    _console.print(table)
    _console.print()


def _save_mcp_profile_entry(record: dict) -> None:
    """把一条 server 记录写入 profile；失败仅告警，绝不阻塞连接主流程。"""
    try:
        mcp_profile.save_entry(record)
    except Exception as exc:
        _console.print(f"  [warning]⚠ 写入 MCP profile 失败：{exc}[/]")


def _remove_mcp_server(name: str) -> None:
    """断开指定 MCP 服务并刷新注册表，同步从 profile 删除。"""
    ok = _mcp_manager().remove(name)
    if ok:
        get_registry().refresh_mcp()
        try:
            mcp_profile.remove_entry(name)
        except Exception:
            pass
    print()
    if ok:
        print(f"  {C.GREEN}✓ 已断开 MCP:{C.RESET} {name}")
    else:
        print(f"  {C.RED}未找到 MCP 服务：{C.RESET}{name}")
    print()

