from __future__ import annotations

import argparse
import atexit
import os
import re
import sys
import unicodedata
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding

from .graph import create_search_assistant
from .mcp import get_manager as _mcp_manager, shutdown_manager as _mcp_shutdown


_console = Console()


# ── Styling ────────────────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ORANGE = "\033[38;5;214m"
    GOLD = "\033[38;5;178m"
    GRAY = "\033[38;5;240m"
    CYAN = "\033[38;5;117m"
    GREEN = "\033[38;5;114m"
    RED = "\033[38;5;203m"


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_HIT_RE = re.compile(r"^\d+\.\s+\*\*", re.MULTILINE)


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _visual_width(s: str) -> int:
    s = _strip_ansi(s)
    w = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        else:
            w += 1
    return w


def _term_width(max_width: int = 72) -> int:
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80
    return max(40, min(cols, max_width))


def _pad(content: str, inner_width: int) -> str:
    pad = inner_width - _visual_width(content)
    return content + " " * max(pad, 0)


# ── Blocks ─────────────────────────────────────────────────────────

def welcome_box() -> None:
    w = _term_width()
    inner = w - 4
    border = "─" * (w - 2)
    lines = [
        f"{C.ORANGE}✻{C.RESET} Welcome to {C.BOLD}AskAnswer{C.RESET}!",
        "",
        f"{C.DIM}输入 {C.RESET}{C.CYAN}/help{C.RESET}{C.DIM} 查看命令，{C.RESET}"
        f"{C.CYAN}/exit{C.RESET}{C.DIM} 退出{C.RESET}",
        "",
        f"{C.DIM}cwd: {Path.cwd()}{C.RESET}",
    ]
    print()
    print(f"{C.ORANGE}╭{border}╮{C.RESET}")
    for line in lines:
        print(f"{C.ORANGE}│{C.RESET} {_pad(line, inner)} {C.ORANGE}│{C.RESET}")
    print(f"{C.ORANGE}╰{border}╯{C.RESET}")


def tips_block() -> None:
    print()
    print(f" {C.BOLD}Tips for getting started:{C.RESET}")
    print()
    print(f" {C.DIM}1.{C.RESET} 提出任何问题，我会进行搜索并整理答案")
    print(f" {C.DIM}2.{C.RESET} 问题越具体，结果越精准")
    print(f" {C.DIM}3.{C.RESET} 输入 {C.CYAN}/help{C.RESET} 查看所有命令")
    print()


def help_block() -> None:
    print()
    print(f" {C.BOLD}Commands{C.RESET}")
    print(f"   {C.CYAN}/help{C.RESET}    显示此帮助")
    print(f"   {C.CYAN}/clear{C.RESET}   清屏并开始新会话")
    print(f"   {C.CYAN}/status{C.RESET}  查看当前会话信息")
    print(f"   {C.CYAN}/mcp{C.RESET}     管理 MCP 服务 ({C.DIM}/mcp 查看子命令{C.RESET})")
    print(f"   {C.CYAN}/exit{C.RESET}    退出 (也可 /quit, /q, Ctrl-D)")
    print()


def mcp_help_block() -> None:
    print()
    print(f" {C.BOLD}/mcp{C.RESET}")
    print(f"   {C.CYAN}/mcp <url> [name]{C.RESET}      连接一个 MCP 服务 (HTTP/SSE)")
    print(f"   {C.CYAN}/mcp list{C.RESET}              列出已连接的 MCP 服务")
    print(f"   {C.CYAN}/mcp tools [server]{C.RESET}    列出工具 (可选按 server 过滤)")
    print(f"   {C.CYAN}/mcp remove <name>{C.RESET}     断开指定服务")
    print()


def status_block(thread_id: str) -> None:
    print()
    print(f" {C.BOLD}Status{C.RESET}")
    print(f"   {C.DIM}thread:{C.RESET}  {thread_id}")
    print(f"   {C.DIM}cwd:{C.RESET}     {Path.cwd()}")
    servers = _mcp_manager().list_servers()
    if servers:
        summary = ", ".join(f"{s['name']}({s['tools']})" for s in servers)
        print(f"   {C.DIM}mcp:{C.RESET}     {summary}")
    else:
        print(f"   {C.DIM}mcp:{C.RESET}     {C.DIM}（未连接，/mcp <url> 添加）{C.RESET}")
    print()


# ── Streaming progress ────────────────────────────────────────────

def _marker(title: str, detail: str = "") -> str:
    body = f"{C.BOLD}{title}{C.RESET}"
    if detail:
        body += f"{C.DIM}({detail}){C.RESET}"
    return f"  {C.ORANGE}⏺{C.RESET} {body}"


def _truncate(s: str, limit: int = 60) -> str:
    s = " ".join(str(s or "").split())
    if _visual_width(s) <= limit:
        return s
    out = ""
    for ch in s:
        if _visual_width(out + ch) > limit - 1:
            break
        out += ch
    return out + "…"


def stream_query(app, query: str, thread_id: str) -> str:
    final_answer = ""
    config = {"configurable": {"thread_id": thread_id}}
    graph_input: object = {"messages": [HumanMessage(content=query)]}

    print()
    while True:
        interrupt_payload = None
        for chunk in app.stream(
            graph_input,
            config=config,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupt_payload = _extract_interrupt_value(update)
                    continue
                if not isinstance(update, dict):
                    continue
                final_answer = _render_node_update(node, update, final_answer)

        if interrupt_payload is None:
            break

        resume_value = _prompt_shell_confirmation(interrupt_payload)
        graph_input = Command(resume=resume_value)

    if not final_answer:
        try:
            state = app.get_state({"configurable": {"thread_id": thread_id}})
            vals = getattr(state, "values", {}) or {}
            final_answer = vals.get("final_answer") or ""
            if not final_answer:
                msgs = vals.get("messages") or []
                if msgs:
                    content = getattr(msgs[-1], "content", "")
                    if isinstance(content, str):
                        final_answer = content
        except Exception:
            pass

    return final_answer or "未生成答案。"


def _render_node_update(node: str, update: dict, final_answer: str) -> str:
    if node == "understand":
        intent = update.get("intent", "")
        if intent == "file_read":
            detail = f"file_read: {_truncate(update.get('file_path', ''))}"
        elif intent == "chat":
            detail = "chat"
        else:
            detail = f"search: {_truncate(update.get('search_query', ''))}"
        print(_marker("Understand", detail))
    elif node == "file_read":
        if update.get("final_answer"):
            final_answer = update["final_answer"]
        print(_marker("FileRead", "读取完成"))
    elif node == "search":
        if update.get("step") == "search_failed":
            print(_marker("Search", "失败，回退到模型知识"))
        else:
            sr = update.get("search_results", "") or ""
            hits = len(_HIT_RE.findall(sr))
            detail = f"Top {hits} 结果" if hits else "完成"
            print(_marker("Search", detail))
    elif node == "answer":
        if update.get("final_answer"):
            final_answer = update["final_answer"]
        print(_marker("Answer", "整合中"))
    elif node == "sorcery":
        if update.get("final_answer"):
            final_answer = update["final_answer"]
        if update.get("step") == "retry_search":
            nsq = _truncate(update.get("search_query", ""))
            print(_marker("Sorcery", f"不够好，重搜：{nsq}"))
        else:
            print(_marker("Sorcery", "通过"))
    elif node == "tools":
        print(_marker("Tools", "执行工具调用"))
    elif node == "shell_plan":
        plans = update.get("pending_shell") or {}
        detail = f"生成 {len(plans)} 条命令" if plans else "规划完成"
        print(_marker("ShellPlan", detail))
    else:
        print(_marker(node))
    return final_answer


def _extract_interrupt_value(update):
    if isinstance(update, (list, tuple)) and update:
        first = update[0]
    else:
        first = update
    return getattr(first, "value", first)


def _prompt_shell_confirmation(payload) -> dict:
    data = payload if isinstance(payload, dict) else {}
    command = data.get("command") or str(payload)
    explanation = data.get("explanation") or ""

    print()
    print(f"  {C.ORANGE}⏸{C.RESET}  {C.BOLD}需要确认 Shell 命令{C.RESET}")
    print(f"    {C.DIM}命令：{C.RESET}{C.CYAN}{command}{C.RESET}")
    if explanation:
        print(f"    {C.DIM}说明：{C.RESET}{explanation}")
    try:
        reply = input(f"    {C.ORANGE}是否执行？(y/N):{C.RESET} ").strip().lower()
    except EOFError:
        reply = ""
    except KeyboardInterrupt:
        reply = ""
        print()
    approve = reply in ("y", "yes")
    # 回传完整信息：批准与否 + 用户当时看到的命令，
    # 让节点在重放时不因为 LLM 再次生成而执行到别的命令。
    return {"approve": approve, "command": command, "explanation": explanation}


# ── Render ────────────────────────────────────────────────────────

def render_answer(answer: str) -> None:
    print()
    _console.print(Padding(Markdown(answer or "_(空答案)_"), (0, 2)))
    print()


def render_error(message: str) -> None:
    print()
    print(f"  {C.RED}✗ 运行失败{C.RESET}")
    print(f"  {C.DIM}{message}{C.RESET}")
    print()


# ── Interactive loop ──────────────────────────────────────────────

def _prompt_boxed_input() -> str:
    w = _term_width()
    border = "─" * (w - 2)
    print(f"{C.ORANGE}╭{border}╮{C.RESET}")
    sys.stdout.write(f"{C.ORANGE}│{C.RESET} {C.ORANGE}>{C.RESET} ")
    sys.stdout.flush()
    try:
        text = input()
    finally:
        print(f"{C.ORANGE}╰{border}╯{C.RESET}")
    return text


def handle_command(cmd: str, *, thread_id: str) -> tuple[bool, str]:
    stripped = cmd.strip()
    head, _, tail = stripped.partition(" ")
    head_lc = head.lower()
    tail = tail.strip()

    if head_lc in {"/exit", "/quit", "/q"}:
        print(f"\n{C.DIM}再见。{C.RESET}")
        return False, thread_id
    if head_lc == "/help":
        help_block()
    elif head_lc == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
        welcome_box()
        thread_id = str(uuid.uuid4())
        print(f"\n  {C.DIM}已开始新会话：{thread_id[:8]}…{C.RESET}\n")
    elif head_lc == "/status":
        status_block(thread_id)
    elif head_lc == "/mcp":
        handle_mcp_command(tail)
    else:
        print(
            f"\n  {C.RED}未知命令：{C.RESET}{stripped}  "
            f"({C.CYAN}/help{C.RESET} 查看可用命令)\n"
        )
    return True, thread_id


def handle_mcp_command(args: str) -> None:
    if not args:
        _print_mcp_servers()
        mcp_help_block()
        return

    first, _, rest = args.partition(" ")
    rest = rest.strip()
    first_lc = first.lower()

    if first_lc in {"list", "ls"}:
        _print_mcp_servers()
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
        _add_mcp_url(first, rest or None)
    else:
        print(
            f"\n  {C.RED}无法识别 /mcp 参数：{C.RESET}{args}\n"
            f"  {C.DIM}提示：URL 需以 http:// 或 https:// 开头{C.RESET}"
        )
        mcp_help_block()


def _add_mcp_url(url: str, name: str | None) -> None:
    try:
        registered = _mcp_manager().add_url(url, name=name)
    except Exception as exc:
        render_error(f"MCP 连接失败: {exc}")
        return
    tools = _mcp_manager().list_tools(server=registered)
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


def _remove_mcp_server(name: str) -> None:
    ok = _mcp_manager().remove(name)
    print()
    if ok:
        print(f"  {C.GREEN}✓ 已断开 MCP:{C.RESET} {name}")
    else:
        print(f"  {C.RED}未找到 MCP 服务：{C.RESET}{name}")
    print()


def _print_mcp_servers() -> None:
    servers = _mcp_manager().list_servers()
    print()
    if not servers:
        print(f"  {C.DIM}（暂未连接 MCP 服务）{C.RESET}")
        print()
        return
    print(f"  {C.BOLD}MCP Servers ({len(servers)}){C.RESET}")
    for s in servers:
        print(
            f"   {C.GREEN}●{C.RESET} {C.BOLD}{s['name']}{C.RESET}  "
            f"{C.DIM}{s['transport']} · {s['tools']} tools{C.RESET}"
        )
        if s.get("url"):
            print(f"     {C.DIM}{s['url']}{C.RESET}")
    print()


def _print_mcp_tools(server: str | None) -> None:
    tools = _mcp_manager().list_tools(server=server)
    print()
    if not tools:
        hint = f"{server} 无工具 / 未连接" if server else "暂无 MCP 工具，使用 /mcp <url> 添加"
        print(f"  {C.DIM}（{hint}）{C.RESET}")
        print()
        return
    scope = f" ({server})" if server else ""
    print(f"  {C.BOLD}MCP Tools{scope} · {len(tools)}{C.RESET}")
    for t in tools:
        desc = _truncate(t.get("description") or "", 56)
        print(f"    {C.CYAN}{t['name']}{C.RESET}  {C.DIM}{desc}{C.RESET}")
    print()


def interactive_loop(app) -> int:
    thread_id = str(uuid.uuid4())

    welcome_box()
    tips_block()

    while True:
        try:
            text = _prompt_boxed_input().strip()
        except EOFError:
            print(f"\n{C.DIM}再见。{C.RESET}")
            return 0
        except KeyboardInterrupt:
            print(f"\n{C.DIM}(已取消，输入 /exit 退出){C.RESET}")
            continue

        if not text:
            continue

        if text.startswith("/"):
            keep_going, thread_id = handle_command(text, thread_id=thread_id)
            if not keep_going:
                return 0
            continue

        try:
            answer = stream_query(app, text, thread_id)
        except Exception as exc:
            render_error(str(exc))
            continue

        render_answer(answer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AskAnswer 命令行工具")
    parser.add_argument("question", nargs="?", help="要提问的内容")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    atexit.register(_mcp_shutdown)

    try:
        app = create_search_assistant()
    except Exception as exc:
        render_error(f"初始化失败: {exc}")
        return 1

    if args.question:
        thread_id = str(uuid.uuid4())
        try:
            answer = stream_query(app, args.question, thread_id)
        except Exception as exc:
            render_error(str(exc))
            return 1
        render_answer(answer)
        return 0

    return interactive_loop(app)


if __name__ == "__main__":
    raise SystemExit(main())
