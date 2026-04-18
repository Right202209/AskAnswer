from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding

from .graph import create_search_assistant


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
    print(f"   {C.CYAN}/exit{C.RESET}    退出 (也可 /quit, /q, Ctrl-D)")
    print()


def status_block(thread_id: str) -> None:
    print()
    print(f" {C.BOLD}Status{C.RESET}")
    print(f"   {C.DIM}thread:{C.RESET}  {thread_id}")
    print(f"   {C.DIM}cwd:{C.RESET}     {Path.cwd()}")
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

    print()
    for chunk in app.stream(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue

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
                    hits = sr.count("**") // 2
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
            else:
                print(_marker(node))

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
    cmd = cmd.lower()
    if cmd in {"/exit", "/quit", "/q"}:
        print(f"\n{C.DIM}再见。{C.RESET}")
        return False, thread_id
    if cmd == "/help":
        help_block()
    elif cmd == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
        welcome_box()
        thread_id = str(uuid.uuid4())
        print(f"\n  {C.DIM}已开始新会话：{thread_id[:8]}…{C.RESET}\n")
    elif cmd == "/status":
        status_block(thread_id)
    else:
        print(
            f"\n  {C.RED}未知命令：{C.RESET}{cmd}  "
            f"({C.CYAN}/help{C.RESET} 查看可用命令)\n"
        )
    return True, thread_id


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
