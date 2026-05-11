# 命令行入口与交互式 REPL。
# 设计要点：
# - 用 ANSI 转义自绘“面板/边框”样式，无第三方 TUI 依赖；
# - app.stream(stream_mode=["updates","messages"]) 同时拿“节点完成事件”
#   和“LLM token 流”，前者打 ⏺ 标记 + 耗时，后者用 rich.live 实时渲染答案；
# - HITL（人机确认）场景下监听 __interrupt__，从 CLI 拿用户输入并 Command(resume=...) 续跑；
# - 输入栏走 prompt_toolkit：历史回溯 / 斜杠补全 / 反斜杠续行 / bottom_toolbar 状态栏；
# - 斜杠命令（/help、/clear、/status、/model、/mcp、/exit）由 handle_command 路由；
# - `!<cmd>` 前缀绕开 LangGraph 直接 shell 执行（带危险检查）。
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shlex
import sys
import time
import unicodedata
import uuid
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)
from langgraph.types import Command
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from .graph import create_search_assistant, draw_search_assistant_mermaid
from .audit import begin_run, end_run, flush_pending, log_event
from .intents import get_intent_registry
from .load import current_model_label, model, set_model
from .mcp import get_manager as _mcp_manager, shutdown_manager as _mcp_shutdown
from .persistence import (
    AuditEvent,
    ThreadMeta,
    get_persistence,
    shutdown_persistence,
)
from .pricing import estimate_cost_usd, format_cost
from .registry import get_registry
from .schema import ContextSchema
from .timetravel import _update_state, fork_thread, list_checkpoints, rewind_to
from .tools import check_dangerous, execute_shell_command, gen_shell_command_spec
from .ui_input import SLASH_COMMANDS, cmd_meta, make_session, read_line
from .ui_select import CANCELLED, select_option
from .ui_spinner import Spinner


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


# 缓存最近一次 ``/threads`` 的结果，``/resume <序号>`` / ``/delete <序号>`` 据此寻址。
# REPL 是单线程的，不需要锁。
_LAST_LIST: list[ThreadMeta] = []


# ── Styling ────────────────────────────────────────────────────────

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


# 用于剥离字符串里的 ANSI 序列以便正确计算可视宽度
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# 识别搜索结果里 “1. **标题**” 这种条目的行首
_HIT_RE = re.compile(r"^\d+\.\s+\*\*", re.MULTILINE)


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


# ── Blocks ─────────────────────────────────────────────────────────

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


def welcome_box() -> None:
    """启动时画的欢迎面板。"""
    body = Group(
        Text.from_markup(
            "[brand]✻[/] Welcome to [bold]AskAnswer[/]!"
        ),
        Text(""),
        Text.from_markup(
            "[subtle]输入[/] [info]/help[/] [subtle]查看命令，[/]"
            "[info]/exit[/] [subtle]退出[/]"
        ),
        Text(""),
        Text.from_markup(f"[subtle]cwd:   {Path.cwd()}[/]"),
        Text.from_markup(f"[subtle]model: {_current_model_name()}[/]"),
    )
    _console.print()
    _console.print(
        Panel(
            body,
            border_style="brand",
            box=box.ROUNDED,
            padding=(0, 2),
            expand=False,
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
                     "/mcp tools my-server", "/mcp remove my-server"],
        "/threads": ["/threads", "/threads sql 关键词"],
        "/resume":  ["/resume 1", "/resume 1e14b9b"],
        "/title":   ["/title 周三的 SQL 调试"],
        "/delete":  ["/delete 1", "/delete 1e14b9b"],
        "/checkpoints": ["/checkpoints"],
        "/undo":    ["/undo", "/undo 2"],
        "/jump":    ["/jump 3"],
        "/fork":    ["/fork", "/fork 2"],
        "/audit":   ["/audit", "/audit 1 --limit 20", "/audit --kind tool_call"],
        "/usage":   ["/usage --days 1", "/usage --thread 1"],
        "/export":  ["/export 1 --format md --out /tmp/thread.md",
                     "/export current --format json --out /tmp/thread.json"],
        "/import":  ["/import /tmp/thread.json"],
    }
    return table.get(cmd, [])


def mcp_help_block() -> None:
    """/mcp 子命令帮助。"""
    print()
    print(f" {C.BOLD}/mcp{C.RESET}")
    print(f"   {C.CYAN}/mcp <url> [name]{C.RESET}      连接一个 MCP 服务 (HTTP/SSE)")
    print(f"   {C.CYAN}/mcp list{C.RESET}              列出已连接的 MCP 服务")
    print(f"   {C.CYAN}/mcp tools [server]{C.RESET}    列出工具 (可选按 server 过滤)")
    print(f"   {C.CYAN}/mcp remove <name>{C.RESET}     断开指定服务")
    print()


def status_block(thread_id: str) -> None:
    """/status 输出：当前线程 ID、CWD、模型、MCP 连接状态、持久化信息。"""
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
        thread_count = len(pm.list_threads(limit=10000))
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


# ── Streaming progress ────────────────────────────────────────────

def _marker(title: str, detail: str = "", elapsed: float | None = None) -> str:
    """生成一行节点进度标记：⏺ Title    detail  · 1.2s。

    用固定宽度的 title 列对齐多行标记；detail 用 dim 字体次级化，
    耗时只在 ≥50ms 时显示（条件路由节点几乎瞬时，打耗时没意义）。
    """
    # 标题列宽 10 足以容下所有已知节点名（Understand / ShellPlan 最长）
    head = f"{C.ORANGE}⏺{C.RESET} {C.BOLD}{title:<10}{C.RESET}"
    parts = [head]
    if detail:
        parts.append(f"{C.DIM}{detail}{C.RESET}")
    if elapsed is not None and elapsed >= 0.05:
        parts.append(f"{C.DIM}· {elapsed:.1f}s{C.RESET}")
    return "  " + "  ".join(parts)


# 节点 → spinner 显示文案的映射；未列出的节点用兜底文案。
_PHASE_TEXT = {
    "understand":  "理解意图…",
    "answer":      "思考中…",
    "tools":       "执行工具…",
    "shell_plan":  "规划 shell 命令…",
    "search":      "联网搜索…",
    "file_read":   "读取文件…",
    "sorcery":     "评估答案质量…",
}


def _phase_text(node: str) -> str:
    return _PHASE_TEXT.get(node, "思考中…")


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


def _runtime_context() -> ContextSchema:
    """从环境变量构造一份 ContextSchema 传给图（CLI 是参数注入的边界）。"""
    return ContextSchema(
        db_dsn=os.getenv("WLANGGRAPH_POSTGRES_DSN") or None,
        db_dialect=os.getenv("ASKANSWER_DB_DIALECT") or None,
        tenant_id=os.getenv("ASKANSWER_TENANT_ID") or None,
    )


def stream_query(
    app,
    query: str,
    thread_id: str,
    runtime_context: ContextSchema | None = None,
) -> str:
    """跑一次完整的图调用：spinner 提示等待 + 实时流式渲染答案 + 处理 HITL。"""
    final_answer = ""
    config = {"configurable": {"thread_id": thread_id}}
    context = runtime_context or _runtime_context()
    graph_input: object = {"messages": [HumanMessage(content=query)]}
    audit_tokens = begin_run(thread_id)
    # 缓存一次最终状态，避免 finally 与下文 meta 持久化各 get_state 一次
    final_state_values: dict | None = None

    print()
    spinner = Spinner("理解意图…")
    spinner.start()
    # 节点之间的耗时：每次收到 update 时算一次差，再把秒表归零。
    last_finish = time.monotonic()

    # rich.Live 在 answer 节点的 LLM token 流到达时启动，承担实时渲染。
    # streamed 标记用来告诉调用者“最终答案已经在屏上了，不要再 render_answer 一次”。
    live_state = {"live": None, "buf": "", "in_tool": False, "streamed": False}

    try:
        while True:
            interrupt_payload = None
            for chunk_mode, payload in app.stream(
                graph_input,
                config=config,
                context=context,
                stream_mode=["updates", "messages"],
            ):
                if chunk_mode == "messages":
                    _handle_message_chunk(payload, spinner, live_state)
                    continue

                # chunk_mode == "updates"
                for node, update in (payload or {}).items():
                    if node == "__interrupt__":
                        interrupt_payload = _extract_interrupt_value(update)
                        continue
                    if not isinstance(update, dict):
                        continue
                    elapsed = time.monotonic() - last_finish
                    last_finish = time.monotonic()
                    final_answer = _on_node_update(
                        node, update, elapsed, final_answer, spinner, live_state,
                    )

            # 兜底：流结束后还可能挂着 interrupt（部分 langgraph 版本不走 __interrupt__）
            if interrupt_payload is None:
                interrupt_payload = _pending_interrupt(app, config)
            if interrupt_payload is None:
                break

            # HITL：暂停 spinner 让用户看清楚要确认的命令；resume 后再启动新一轮。
            spinner.stop()
            _close_live(live_state)
            resume_value = _prompt_shell_confirmation(interrupt_payload)
            graph_input = Command(resume=resume_value)
            spinner = Spinner("继续执行…")
            spinner.start()
            last_finish = time.monotonic()
    finally:
        _close_live(live_state)
        spinner.stop()
        audit_intent = None
        try:
            audit_state = app.get_state(config)
            final_state_values = getattr(audit_state, "values", {}) or {}
            audit_intent = final_state_values.get("intent")
        except Exception:
            pass
        flush_pending(thread_id=thread_id, intent=audit_intent)
        end_run(audit_tokens)

    # 兜底：若节点流里没拿到 final_answer，从 state 里找最后一条消息内容
    if not final_answer:
        try:
            vals = final_state_values
            if vals is None:
                state = app.get_state({"configurable": {"thread_id": thread_id}})
                vals = getattr(state, "values", {}) or {}
                final_state_values = vals
            final_answer = vals.get("final_answer") or ""
            if not final_answer:
                msgs = vals.get("messages") or []
                if msgs:
                    content = getattr(msgs[-1], "content", "")
                    if isinstance(content, str):
                        final_answer = content
        except Exception:
            pass

    # Live 已经把答案画在屏上了，就只补一个空行做间距；
    # 没走 Live 的（典型场景：模型不支持 stream，或 sorcery 重写答案）才走传统渲染。
    if live_state.get("streamed"):
        print()
    else:
        render_answer(final_answer or "未生成答案。")

    # 持久化线程元数据：每次问答后写一行（首次写入会自动取 preview 前 30 字符做 title）。
    # 失败不影响主流程 —— 用户拿到回答比记账更重要。
    try:
        meta_values = final_state_values
        if meta_values is None:
            meta_state = app.get_state(config)
            meta_values = getattr(meta_state, "values", {}) or {}
        msgs = meta_values.get("messages") or []
        human_count = sum(1 for m in msgs if isinstance(m, HumanMessage))
        preview_text = (query or "").strip().replace("\n", " ")[:80] or None
        get_persistence().upsert_meta(
            thread_id,
            intent=meta_values.get("intent"),
            model_label=current_model_label(),
            preview=preview_text,
            message_count=human_count,
        )
    except Exception:
        pass

    return final_answer or "未生成答案。"


def _handle_message_chunk(payload, spinner: Spinner, live_state: dict) -> None:
    """处理 stream_mode='messages' 通道的 token 增量。

    策略：
    - 只关心 ``AIMessageChunk`` —— ToolMessage / HumanMessage 直接忽略；
    - 看到 ``tool_call_chunks``（LLM 正在生成工具调用）时把 ``in_tool`` 置位，
      下一次出现 user-facing content 时清空 buffer，让“工具路由阶段的胡言乱语”
      不污染最终答案的渲染；
    - 首次 user-facing content 出现时切到 rich.Live：spinner 让位、Markdown 实时刷新。
    """
    if not isinstance(payload, tuple) or len(payload) != 2:
        return
    msg, _meta = payload
    if not isinstance(msg, AIMessageChunk):
        return

    # tool_call_chunks 出现 = LLM 正在“规划工具调用”。Mark phase；下一段 content 来时重置 buffer。
    if getattr(msg, "tool_call_chunks", None):
        live_state["in_tool"] = True
        return

    content = msg.content if isinstance(msg.content, str) else ""
    if not content:
        return

    if live_state["in_tool"]:
        live_state["buf"] = ""
        live_state["in_tool"] = False

    if live_state["live"] is None:
        # 第一次拿到 user-facing token：让 spinner 让位，启动 Live 渲染。
        # 在 Live 之前插一条 Rule，把"进度 trace"与"答案正文"在视觉上拆开。
        spinner.stop()
        _console.print()
        _console.print(Rule(title="[subtle]Answer[/]", style="muted", align="left"))
        live = Live(
            Padding(Markdown(""), (0, 2)),
            console=_console,
            refresh_per_second=15,
            transient=False,
        )
        live.start()
        live_state["live"] = live
        live_state["streamed"] = True

    live_state["buf"] += content
    live_state["live"].update(Padding(Markdown(live_state["buf"]), (0, 2)))


def _on_node_update(
    node: str,
    update: dict,
    elapsed: float,
    final_answer: str,
    spinner: Spinner,
    live_state: dict,
) -> str:
    """节点完成时被调：打 ⏺ 标记 + 维护 spinner / Live 生命周期。

    返回新的 ``final_answer``（沿用旧的或被节点更新覆盖）。
    """
    # 父图的 "answer" 节点完成（react 子图整段跑完）：把 Live 收尾、答案确权。
    if node == "answer":
        if update.get("final_answer"):
            final_answer = update["final_answer"]
        if live_state["live"] is not None:
            # 用权威 final_answer 替换 Live 的当前内容，防止 buffer 与最终答案有偏差
            if final_answer:
                live_state["live"].update(
                    Padding(Markdown(final_answer), (0, 2))
                )
            _close_live(live_state)
        spinner.freeze_for(lambda: print(_marker("Answer", "完成", elapsed)))
        spinner.transition(_phase_text("sorcery"))
        return final_answer

    # 其它节点：一律先暂停 spinner 写一行 ⏺ 标记，再切到下一阶段文案。
    new_final = _render_node_update_safely(
        node, update, final_answer, elapsed, spinner,
    )
    spinner.transition(_phase_text(node))
    return new_final


def _render_node_update_safely(
    node: str, update: dict, final_answer: str, elapsed: float, spinner: Spinner,
) -> str:
    """``spinner.freeze_for`` 不能跨返回值传递，这里包一层用列表传出。"""
    holder = [final_answer]

    def _do():
        holder[0] = _render_node_update(node, update, final_answer, elapsed)

    spinner.freeze_for(_do)
    return holder[0]


def _close_live(live_state: dict) -> None:
    """关闭 rich.Live，保留已渲染内容（transient=False）。"""
    live = live_state.get("live")
    if live is not None:
        live.stop()
    live_state["live"] = None
    # buffer 不清，保留给下游 final_answer 兜底用


def _render_node_update(
    node: str, update: dict, final_answer: str, elapsed: float | None = None,
) -> str:
    """根据节点名渲染对应的进度标记，并把 final_answer 顺手记下来。"""
    if node == "understand":
        detail = _truncate(_intent_cli_label(update))
        print(_marker("Understand", detail, elapsed))
    elif node == "file_read":
        # （兼容老拓扑）file_read 节点已合并到 react 的 read_file 工具
        if update.get("final_answer"):
            final_answer = update["final_answer"]
        print(_marker("FileRead", "读取完成", elapsed))
    elif node == "search":
        # （兼容老拓扑）search 作为独立节点的版本
        if update.get("step") == "search_failed":
            print(_marker("Search", "失败，回退到模型知识", elapsed))
        else:
            sr = update.get("search_results", "") or ""
            hits = len(_HIT_RE.findall(sr))
            detail = f"Top {hits} 结果" if hits else "完成"
            print(_marker("Search", detail, elapsed))
    elif node == "answer":
        # 注意：父图 "answer" 节点在 stream_query 已被特别处理（含 Live 收尾）；
        # 这里只是兼容旧调用路径。
        if update.get("final_answer"):
            final_answer = update["final_answer"]
        print(_marker("Answer", "整合中", elapsed))
    elif node == "sorcery":
        if update.get("final_answer"):
            final_answer = update["final_answer"]
        if update.get("step") == "retry_search":
            directive = update.get("retry_directive") or {}
            nsq = _truncate(update.get("search_query", "") or directive.get("instruction", ""))
            print(_marker("Sorcery", f"不够好，重搜：{nsq}", elapsed))
        else:
            print(_marker("Sorcery", "通过", elapsed))
    elif node == "tools":
        print(_marker("Tools", "执行工具调用", elapsed))
    elif node == "shell_plan":
        plans = update.get("pending_shell") or {}
        detail = f"生成 {len(plans)} 条命令" if plans else "规划完成"
        print(_marker("ShellPlan", detail, elapsed))
    else:
        # 兜底：未知节点也给一行标记，至少能看到流转
        print(_marker(node, "", elapsed))
    return final_answer


def _intent_cli_label(update: dict) -> str:
    return get_intent_registry().get(update.get("intent")).cli_label(update)


def _extract_interrupt_value(update):
    """LangGraph 不同版本里 __interrupt__ 的载荷形态不同，做一层兼容。"""
    if isinstance(update, (list, tuple)) and update:
        first = update[0]
    else:
        first = update
    return getattr(first, "value", first)


def _pending_interrupt(app, config):
    """从 state.tasks 里反查是否还有挂起的 interrupt，作为兜底。"""
    try:
        snapshot = app.get_state(config)
    except Exception:
        return None
    tasks = getattr(snapshot, "tasks", None) or ()
    for task in tasks:
        interrupts = getattr(task, "interrupts", None) or ()
        if interrupts:
            first = interrupts[0]
            return getattr(first, "value", first)
    return None


def _prompt_shell_confirmation(payload) -> dict:
    """用上下选项菜单提示用户确认 shell 命令；选项含 执行 / 取消 / 补充说明后重新生成。"""
    data = payload if isinstance(payload, dict) else {}
    command = data.get("command") or str(payload)
    explanation = data.get("explanation") or ""
    instruction = data.get("instruction") or ""

    # 循环：用户选 “补充说明后重新生成” 时，重新生成命令并再次询问
    while True:
        print()
        print(f"  {C.ORANGE}⏸{C.RESET}  {C.BOLD}需要确认 Shell 命令{C.RESET}")
        print(f"    {C.DIM}命令：{C.RESET}{C.CYAN}{command}{C.RESET}")
        if explanation:
            print(f"    {C.DIM}说明：{C.RESET}{explanation}")
        # 默认光标停在“取消”：避免误回车直接执行高风险命令。
        idx, _ = select_option(
            ["执行", "取消", "补充说明后重新生成"],
            prompt="选择操作（↑/↓ 导航 · Enter 确认）：",
            default=1,
        )
        if idx == 0:
            return {"approve": True, "command": command, "explanation": explanation}
        if idx == 2:
            more = _read_more_prompt()
            if not more:
                print(f"    {C.DIM}未输入补充说明，保持原命令。{C.RESET}")
                continue
            combined = (
                f"{instruction}\n补充说明：{more}".strip()
                if instruction else more
            )
            try:
                new_command, new_explanation = gen_shell_command_spec(combined)
            except Exception as exc:
                print(f"    {C.RED}生成失败：{C.RESET}{exc}")
                continue
            if not new_command:
                print(f"    {C.RED}未能生成有效命令，保持原命令。{C.RESET}")
                continue
            command = new_command
            explanation = new_explanation
            instruction = combined
            continue
        # idx == 1（取消）或 CANCELLED（Esc/Ctrl-C）一律视为不执行
        return {"approve": False, "command": command, "explanation": explanation}


def _read_more_prompt() -> str:
    """让用户输入“补充说明”一行，Ctrl-C/D 视为放弃补充。"""
    try:
        return input(f"    {C.GOLD}补充说明:{C.RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


# ── Render ────────────────────────────────────────────────────────

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


# ── Bang shell shortcut ───────────────────────────────────────────

def run_bang_command(command: str) -> None:
    """`!<cmd>` 走绕过 LangGraph 的直执行路径；危险命令仍要二次确认。"""
    command = command.strip()
    if not command:
        # 空命令：打印用法
        print()
        print(f"  {C.DIM}用法：{C.RESET}{C.CYAN}!<shell command>{C.RESET}  "
              f"{C.DIM}例：!ls -la{C.RESET}")
        print()
        return

    # 命中危险模式：必须用户显式确认
    danger = check_dangerous(command)
    if danger:
        print()
        print(f"  {C.RED}⚠ 高风险命令（{danger}）{C.RESET}")
        print(f"    {C.DIM}命令：{C.RESET}{C.CYAN}{command}{C.RESET}")
        try:
            reply = input(f"    {C.ORANGE}仍然执行? (y/N):{C.RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            reply = ""
        if reply not in ("y", "yes"):
            print(f"    {C.DIM}已取消。{C.RESET}")
            print()
            return

    # bang 模式 shell=True：保留管道、重定向等用户自己手敲的语法
    output = execute_shell_command(command, shell=True)

    print()
    for line in output.splitlines():
        print(f"    {line}")
    print()


# ── Interactive loop ──────────────────────────────────────────────

def _build_status_provider(thread_box: list[str]):
    """构造给 ``ui_input.make_session`` 用的 status 回调。

    ``thread_box`` 是单元素列表，作为对当前 ``thread_id`` 的可变引用
    （``/clear`` / ``/resume`` / ``/delete`` 切换会话时直接改写 ``thread_box[0]``）。
    """
    def get_status() -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = [
            ("thread", (thread_box[0] or "?")[:8]),
            ("model",  current_model_label() or "—"),
        ]
        try:
            servers = _mcp_manager().list_servers()
            if servers:
                items.append(("mcp", str(len(servers))))
        except Exception:
            pass
        return items

    return get_status


def _draw_top_border() -> None:
    """与 prompt_toolkit 输入区配套的视觉上边框；下边框由 ``_draw_bottom_border`` 收尾。"""
    w = _term_width()
    border = "─" * (w - 2)
    print(f"{C.ORANGE}╭{border}╮{C.RESET}")


def _draw_bottom_border() -> None:
    w = _term_width()
    border = "─" * (w - 2)
    print(f"{C.ORANGE}╰{border}╯{C.RESET}")


def handle_command(cmd: str, *, thread_id: str, app=None) -> tuple[bool, str]:
    """斜杠命令路由。返回 (是否继续运行, 当前 thread_id)。

    ``app`` 参数：``/resume`` 需要查询 ``app.get_state`` 检测挂起的 interrupt；
    其它命令不强依赖。
    """
    stripped = cmd.strip()
    head, _, tail = stripped.partition(" ")
    head_lc = head.lower()
    tail = tail.strip()

    if head_lc in {"/exit", "/quit", "/q"}:
        print(f"\n{C.DIM}再见。{C.RESET}")
        return False, thread_id
    if head_lc == "/help":
        help_block(tail or None)
    elif head_lc == "/clear":
        # 清屏 + 新建一个 thread_id：等价于一段全新对话。旧线程仍保留在 SqliteSaver 里，
        # 用户可用 /threads 查看 / /resume 恢复 / /delete 删除。
        old_short = thread_id[:8]
        os.system("cls" if os.name == "nt" else "clear")
        welcome_box()
        thread_id = str(uuid.uuid4())
        print(
            f"\n  {C.DIM}已开始新会话：{thread_id[:8]}…  "
            f"上一段保留为 {old_short}…（/threads 查看 · /delete 删除）{C.RESET}\n"
        )
    elif head_lc == "/status":
        status_block(thread_id)
    elif head_lc == "/model":
        handle_model_command(tail, thread_id=thread_id)
    elif head_lc == "/mcp":
        handle_mcp_command(tail, thread_id=thread_id)
    elif head_lc == "/threads":
        handle_threads_command(tail, current=thread_id)
    elif head_lc == "/resume":
        new_id = handle_resume_command(tail, current=thread_id, app=app)
        if new_id:
            thread_id = new_id
    elif head_lc == "/title":
        handle_title_command(tail, thread_id=thread_id)
    elif head_lc == "/delete":
        new_id = handle_delete_command(tail, current=thread_id)
        if new_id:
            # 删的是当前 thread：自动开新会话
            thread_id = new_id
    elif head_lc == "/checkpoints":
        handle_checkpoints_command(tail, thread_id=thread_id, app=app)
    elif head_lc == "/undo":
        handle_undo_command(tail, thread_id=thread_id, app=app)
    elif head_lc == "/jump":
        handle_jump_command(tail, thread_id=thread_id, app=app)
    elif head_lc == "/fork":
        new_id = handle_fork_command(tail, current=thread_id, app=app)
        if new_id:
            thread_id = new_id
    elif head_lc == "/audit":
        handle_audit_command(tail, current=thread_id)
    elif head_lc == "/usage":
        handle_usage_command(tail, current=thread_id)
    elif head_lc == "/export":
        handle_export_command(tail, current=thread_id, app=app)
    elif head_lc == "/import":
        new_id = handle_import_command(tail, app=app)
        if new_id:
            thread_id = new_id
    else:
        print(
            f"\n  {C.RED}未知命令：{C.RESET}{stripped}  "
            f"({C.CYAN}/help{C.RESET} 查看可用命令)\n"
        )
    return True, thread_id


def handle_model_command(args: str, *, thread_id: str) -> None:
    """/model：无参数显示当前模型；带参数尝试切换模型。"""
    if not args:
        print()
        print(f"  {C.BOLD}Model{C.RESET}")
        print(f"   {C.DIM}current:{C.RESET} {current_model_label()}")
        print(f"   {C.DIM}usage:{C.RESET}   {C.CYAN}/model <name>{C.RESET} "
              f"{C.DIM}或{C.RESET} {C.CYAN}/model <provider:name>{C.RESET}")
        print()
        return

    try:
        # set_model 是热替换，所有已 import 的 model 引用都会自动指向新模型
        label = set_model(args)
    except Exception as exc:
        render_error(f"模型切换失败: {exc}")
        return
    log_event(
        kind="model_swap",
        thread_id=thread_id,
        model_label=label,
        args_summary=args,
        immediate=True,
    )

    print()
    print(f"  {C.GREEN}✓ 已切换模型:{C.RESET} {C.BOLD}{label}{C.RESET}")
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


def _remove_mcp_server(name: str) -> None:
    """断开指定 MCP 服务并刷新注册表。"""
    ok = _mcp_manager().remove(name)
    if ok:
        get_registry().refresh_mcp()
    print()
    if ok:
        print(f"  {C.GREEN}✓ 已断开 MCP:{C.RESET} {name}")
    else:
        print(f"  {C.RED}未找到 MCP 服务：{C.RESET}{name}")
    print()


def _print_mcp_servers() -> None:
    """打印已连接的 MCP 服务清单。"""
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
    table.add_column("url", style="subtle", no_wrap=True, overflow="ellipsis", max_width=46)
    for s in servers:
        table.add_row(
            "[success]●[/]",
            s["name"],
            s["transport"],
            str(s["tools"]),
            s.get("url") or "",
        )
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


# ── Threads / Resume / Title / Delete ─────────────────────────────────

def _format_ts(ts: int) -> str:
    """把 epoch 秒格式化为 ``MM-DD HH:MM`` 紧凑形式（同一年）。"""
    if not ts:
        return "—"
    import time as _t
    return _t.strftime("%m-%d %H:%M", _t.localtime(ts))


def _resolve_thread(arg: str) -> ThreadMeta | None:
    """把 ``/resume 1`` 或 ``/resume <id 前缀>`` 解析为 ``ThreadMeta``。

    解析顺序：
    1. 纯数字 → 视作针对最近一次 ``/threads`` 列表的序号；
    2. 完整 ID 精确匹配；
    3. 4 字符及以上前缀匹配（多于一条则返回 None 让上层提示歧义）。
    """
    arg = (arg or "").strip()
    if not arg:
        return None
    pm = get_persistence()

    # 1) 序号：相对最近一次 /threads 的输出
    if arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(_LAST_LIST):
            return _LAST_LIST[idx]
        return None

    # 2) 完整匹配（UUID 是 36 字符，但允许任意完整 ID）
    meta = pm.get_meta(arg)
    if meta is not None:
        return meta

    # 3) 前缀匹配：≥4 字符才生效，避免 "a" 这种过宽匹配
    matches = pm.find_by_prefix(arg, limit=2)
    if len(matches) == 1:
        return matches[0]
    return None


def _has_pending_interrupt(app, thread_id: str) -> bool:
    """检测某 thread 是否还有挂起的 ``interrupt()`` 任务（shell HITL 没收尾）。"""
    if app is None:
        return False
    try:
        snap = app.get_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        return False
    for task in (getattr(snap, "tasks", None) or ()):
        if getattr(task, "interrupts", None):
            return True
    return False


def handle_threads_command(args: str, *, current: str) -> None:
    """``/threads [keyword]``：按 updated_at 倒序列出最近 50 条。"""
    global _LAST_LIST
    keyword = args.strip() or None
    try:
        threads = get_persistence().list_threads(limit=50, query=keyword)
    except Exception as exc:
        render_error(f"读取持久化失败: {exc}")
        return
    _LAST_LIST = threads

    _console.print()
    if not threads:
        hint = f"无匹配 '{keyword}'" if keyword else "暂无历史会话（先聊几句吧）"
        _console.print(f"  [subtle]（{hint}）[/]")
        _console.print()
        return

    title_word = f"  [subtle]· 关键词: {keyword}[/]" if keyword else ""
    _console.print(f"  [bold]Threads ({len(threads)})[/]{title_word}")

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="subtle",
        border_style="muted",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("", width=1, no_wrap=True)
    table.add_column("#", justify="right", style="subtle", no_wrap=True, width=3)
    table.add_column("id", style="info", no_wrap=True, width=8)
    table.add_column("updated", style="subtle", no_wrap=True, width=11)
    table.add_column("intent", style="subtle", no_wrap=True, width=8)
    table.add_column("msgs", justify="right", style="subtle", no_wrap=True, width=4)
    table.add_column("title / preview", no_wrap=True, overflow="ellipsis", ratio=1)

    for i, m in enumerate(threads, 1):
        marker = "[success]●[/]" if m.thread_id == current else " "
        text = (m.title or m.preview or "(空)").strip().replace("\n", " ")
        intent = (m.last_intent or "—")[:8]
        table.add_row(
            marker,
            str(i),
            m.thread_id[:8],
            _format_ts(m.updated_at),
            intent,
            f"{m.message_count}",
            text,
        )
    _console.print(table)
    _console.print(
        "  [subtle]用法：[/]"
        "[info]/resume <序号|id>[/][subtle] 恢复 · [/]"
        "[info]/title <名字>[/][subtle] 命名当前 · [/]"
        "[info]/delete <序号|id>[/][subtle] 删除[/]"
    )
    _console.print()


def handle_resume_command(args: str, *, current: str, app=None) -> str | None:
    """``/resume <序号|id 前缀>``：切换 thread_id 到目标会话。

    返回新的 thread_id；解析失败或用户取消时返回 ``None``。
    """
    if not args.strip():
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/resume <序号|id 前缀>{C.RESET}\n")
        return None

    target = _resolve_thread(args)
    if target is None:
        print(
            f"\n  {C.RED}找不到匹配的会话：{C.RESET}{args}  "
            f"{C.DIM}（先 /threads 看一下序号或 id 前缀）{C.RESET}\n"
        )
        return None

    if target.thread_id == current:
        print(f"\n  {C.DIM}已经在该会话上了：{target.thread_id[:8]}…{C.RESET}\n")
        return None

    # 关键风险：目标会话上次卡在 shell HITL 没 resume；提醒但不阻断
    if _has_pending_interrupt(app, target.thread_id):
        print()
        print(
            f"  {C.GOLD}⚠ 该会话上次中断在 shell 确认未完成。{C.RESET}\n"
            f"    {C.DIM}下一条问题会作为新一轮开始；挂起的确认会被 LangGraph 保留，"
            f"行为可能怪异。{C.RESET}\n"
            f"    {C.DIM}如需先恢复挂起项，请取消本次切换并直接在原会话回答 y/N。{C.RESET}"
        )
        try:
            reply = input(f"    {C.ORANGE}仍要切换？(y/N):{C.RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            reply = ""
        if reply not in ("y", "yes"):
            print(f"    {C.DIM}已取消。{C.RESET}\n")
            return None

    label = target.title or target.preview or "(无标题)"
    print()
    print(
        f"  {C.GREEN}✓ 已切换到会话:{C.RESET} "
        f"{C.CYAN}{target.thread_id[:8]}{C.RESET}  "
        f"{C.DIM}{_truncate(label, 50)}{C.RESET}"
    )
    print(f"  {C.DIM}下一条问题会接续这段会话的历史。{C.RESET}\n")
    return target.thread_id


def handle_title_command(args: str, *, thread_id: str) -> None:
    """``/title <name>``：给当前会话命名。"""
    title = args.strip()
    if not title:
        try:
            meta = get_persistence().get_meta(thread_id)
        except Exception:
            meta = None
        current = meta.title if meta else None
        print()
        print(f"  {C.BOLD}Title{C.RESET}")
        print(f"   {C.DIM}current:{C.RESET} {current or '(未命名)'}")
        print(f"   {C.DIM}usage:{C.RESET}   {C.CYAN}/title <名字>{C.RESET}")
        print()
        return

    try:
        ok = get_persistence().set_title(thread_id, title)
    except Exception as exc:
        render_error(f"重命名失败: {exc}")
        return

    print()
    if ok:
        print(f"  {C.GREEN}✓ 已命名:{C.RESET} {title}")
    else:
        # 没行被更新：通常是当前会话还没产生任何 final_answer，meta 行尚未写入
        print(
            f"  {C.GOLD}⚠ 当前会话尚未持久化{C.RESET}  "
            f"{C.DIM}（先问一个问题让 thread_meta 写入，再 /title 重命名）{C.RESET}"
        )
    print()


def handle_delete_command(args: str, *, current: str) -> str | None:
    """``/delete <序号|id 前缀>``：删除 thread（含 checkpoints + thread_meta）。

    若删除的是 *当前* thread，返回一个新生成的 thread_id 让 REPL 切换；
    其它情况返回 ``None``。
    """
    if not args.strip():
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/delete <序号|id 前缀>{C.RESET}\n")
        return None

    target = _resolve_thread(args)
    if target is None:
        print(
            f"\n  {C.RED}找不到匹配的会话：{C.RESET}{args}  "
            f"{C.DIM}（先 /threads 看一下序号或 id 前缀）{C.RESET}\n"
        )
        return None

    label = target.title or target.preview or "(无标题)"
    is_current = target.thread_id == current
    print()
    print(f"  {C.RED}⚠  即将删除会话{C.RESET}")
    print(
        f"    {C.DIM}id:{C.RESET}    {C.CYAN}{target.thread_id[:8]}{C.RESET}  "
        f"{C.DIM}{_truncate(label, 50)}{C.RESET}"
    )
    if is_current:
        print(f"    {C.GOLD}这是当前会话；删除后将自动开始新会话。{C.RESET}")
    print(f"    {C.DIM}操作不可撤销（同步清除 checkpoints + thread_meta + audit）。{C.RESET}")
    try:
        reply = input(f"    {C.ORANGE}确认删除? (y/N):{C.RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        reply = ""
    if reply not in ("y", "yes"):
        print(f"    {C.DIM}已取消。{C.RESET}\n")
        return None

    try:
        ok = get_persistence().delete_thread(target.thread_id)
    except Exception as exc:
        render_error(f"删除失败: {exc}")
        return None

    # _LAST_LIST 里这一条已失效，做个简单清理避免 /resume 误中
    global _LAST_LIST
    _LAST_LIST = [m for m in _LAST_LIST if m.thread_id != target.thread_id]

    print()
    if ok:
        print(f"  {C.GREEN}✓ 已删除:{C.RESET} {target.thread_id[:8]}…")
    else:
        print(f"  {C.GOLD}⚠ thread_meta 中未找到该 ID（可能 checkpoint 已清）{C.RESET}")
    print()

    if is_current:
        new_id = str(uuid.uuid4())
        print(f"  {C.DIM}已开始新会话：{new_id[:8]}…{C.RESET}\n")
        return new_id
    return None


# ── Checkpoints / Audit / Import-Export ───────────────────────────────

def handle_checkpoints_command(args: str, *, thread_id: str, app=None) -> None:
    if app is None:
        render_error("/checkpoints 只能在已初始化的图上使用")
        return
    try:
        checkpoints = list_checkpoints(app, thread_id)
    except Exception as exc:
        render_error(f"读取 checkpoints 失败: {exc}")
        return

    _console.print()
    if not checkpoints:
        _console.print("  [subtle]（当前会话暂无 checkpoints）[/]")
        _console.print()
        return
    _console.print(f"  [bold]Checkpoints ({len(checkpoints)})[/]")

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="subtle",
        border_style="muted",
        padding=(0, 1),
        expand=False,
    )
    table.add_column("#", justify="right", style="subtle", no_wrap=True)
    table.add_column("node", style="info", no_wrap=True)
    table.add_column("created", style="subtle", no_wrap=True)
    table.add_column("msgs", justify="right", style="subtle", no_wrap=True)
    table.add_column("step", style="subtle", no_wrap=True)
    table.add_column("flags", no_wrap=True)

    for cp in checkpoints:
        flags = []
        if cp.index == 0:
            flags.append("[success]latest[/]")
        if cp.pending_shell:
            flags.append("[warning]pending-shell[/]")
        table.add_row(
            str(cp.index),
            cp.node,
            _format_ts(cp.created_at),
            str(cp.message_count),
            cp.step,
            " ".join(flags),
        )
    _console.print(table)
    _console.print(
        "  [subtle]用法：[/]"
        "[info]/undo [n][/][subtle] 回到第 n 个历史点 · [/]"
        "[info]/jump <index>[/][subtle] 显式跳转 · [/]"
        "[info]/fork [index][/][subtle] 分叉新会话[/]"
    )
    _console.print()


def handle_undo_command(args: str, *, thread_id: str, app=None) -> None:
    raw = args.strip()
    index = 1 if not raw else _parse_nonnegative_int(raw)
    if index is None or index < 1:
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/undo [n]{C.RESET}  {C.DIM}n 默认为 1{C.RESET}\n")
        return
    _rewind_command(app, thread_id, index, label="/undo")


def handle_jump_command(args: str, *, thread_id: str, app=None) -> None:
    index = _parse_nonnegative_int(args.strip())
    if index is None:
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/jump <checkpoint-index>{C.RESET}\n")
        return
    _rewind_command(app, thread_id, index, label="/jump")


def _rewind_command(app, thread_id: str, index: int, *, label: str) -> None:
    if app is None:
        render_error(f"{label} 只能在已初始化的图上使用")
        return
    try:
        target = rewind_to(app, thread_id, index)
    except Exception as exc:
        render_error(f"{label} 失败: {exc}")
        return
    print()
    print(
        f"  {C.GREEN}✓ 已回到 checkpoint:{C.RESET} "
        f"{C.CYAN}#{target.index}{C.RESET}  "
        f"{C.DIM}{target.node} · {target.message_count} msgs · step={target.step}{C.RESET}"
    )
    print(f"  {C.DIM}下一条问题会基于该快照继续。{C.RESET}\n")


def handle_fork_command(args: str, *, current: str, app=None) -> str | None:
    if app is None:
        render_error("/fork 只能在已初始化的图上使用")
        return None
    raw = args.strip()
    index = 0 if not raw else _parse_nonnegative_int(raw)
    if index is None:
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/fork [checkpoint-index]{C.RESET}\n")
        return None
    try:
        new_id = fork_thread(app, current, get_persistence(), index=index)
    except Exception as exc:
        render_error(f"分叉失败: {exc}")
        return None
    print()
    print(
        f"  {C.GREEN}✓ 已分叉新会话:{C.RESET} "
        f"{C.CYAN}{new_id[:8]}{C.RESET}  {C.DIM}来源 checkpoint #{index}{C.RESET}"
    )
    print(f"  {C.DIM}已切换到新会话；旧会话保持不变。{C.RESET}\n")
    return new_id


def handle_audit_command(args: str, *, current: str) -> None:
    parts = _split_args(args)
    if parts is None:
        return
    target_arg = None
    kind = None
    limit = 30
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "--kind" and i + 1 < len(parts):
            kind = parts[i + 1]
            i += 2
        elif part == "--limit" and i + 1 < len(parts):
            parsed = _parse_nonnegative_int(parts[i + 1])
            if parsed is None or parsed <= 0:
                print(f"\n  {C.RED}--limit 必须是正整数{C.RESET}\n")
                return
            limit = parsed
            i += 2
        elif part.startswith("--"):
            print(f"\n  {C.RED}未知参数：{C.RESET}{part}\n")
            return
        elif target_arg is None:
            target_arg = part
            i += 1
        else:
            print(f"\n  {C.RED}多余参数：{C.RESET}{part}\n")
            return
    target = _resolve_thread_or_current(target_arg, current)
    if target is None:
        print(f"\n  {C.RED}找不到匹配的会话：{C.RESET}{target_arg}\n")
        return
    try:
        events = get_persistence().list_audit_events(
            thread_id=target.thread_id,
            kind=kind,
            limit=limit,
        )
    except Exception as exc:
        render_error(f"读取审计失败: {exc}")
        return
    _print_audit_events(events, target)


def handle_usage_command(args: str, *, current: str) -> None:
    parts = _split_args(args)
    if parts is None:
        return
    days = 7
    thread = None
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "--days" and i + 1 < len(parts):
            parsed = _parse_nonnegative_int(parts[i + 1])
            if parsed is None:
                print(f"\n  {C.RED}--days 必须是整数{C.RESET}\n")
                return
            days = parsed
            i += 2
        elif part == "--thread" and i + 1 < len(parts):
            thread = _resolve_thread_or_current(parts[i + 1], current)
            if thread is None:
                print(f"\n  {C.RED}找不到匹配的会话：{C.RESET}{parts[i + 1]}\n")
                return
            i += 2
        else:
            print(f"\n  {C.RED}未知参数：{C.RESET}{part}\n")
            return
    try:
        summary = get_persistence().usage_summary(
            thread_id=thread.thread_id if thread else None,
            days=days,
        )
    except Exception as exc:
        render_error(f"读取 usage 失败: {exc}")
        return
    _print_usage(summary, days=days, thread=thread)


def handle_export_command(args: str, *, current: str, app=None) -> None:
    if app is None:
        render_error("/export 只能在已初始化的图上使用")
        return
    parsed = _parse_export_args(args, current=current)
    if parsed is None:
        return
    target, fmt, out_path = parsed
    try:
        state = app.get_state({"configurable": {"thread_id": target.thread_id}})
        values = getattr(state, "values", {}) or {}
        messages = list(values.get("messages") or [])
        events = get_persistence().list_audit_events(
            thread_id=target.thread_id,
            limit=1000,
        )
    except Exception as exc:
        render_error(f"导出失败: {exc}")
        return

    if fmt == "json":
        payload = _thread_export_payload(target, values, messages, events)
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    else:
        text = _thread_export_markdown(target, messages, events)

    if out_path is None:
        suffix = "json" if fmt == "json" else "md"
        out_path = Path.cwd() / f"askanswer-{target.thread_id[:8]}.{suffix}"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    except Exception as exc:
        render_error(f"写入导出文件失败: {exc}")
        return
    print()
    print(f"  {C.GREEN}✓ 已导出:{C.RESET} {out_path}")
    print()


def handle_import_command(args: str, *, app=None) -> str | None:
    if app is None:
        render_error("/import 只能在已初始化的图上使用")
        return None
    parts = _split_args(args)
    if not parts:
        print(f"\n  {C.RED}用法：{C.RESET}{C.CYAN}/import <path.json>{C.RESET}\n")
        return None
    path = Path(parts[0]).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = messages_from_dict(payload.get("messages") or [])
    except Exception as exc:
        render_error(f"导入失败: {exc}")
        return None
    values = dict(payload.get("values") or {})
    values["messages"] = messages
    values["pending_shell"] = {}
    new_id = str(uuid.uuid4())
    try:
        _update_state(app, {"configurable": {"thread_id": new_id}}, values)
        meta = payload.get("meta") or {}
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        preview = _latest_human_preview(messages)
        get_persistence().upsert_meta(
            new_id,
            title=f"[imported] {meta.get('title') or preview or path.name}"[:120],
            intent=values.get("intent") or meta.get("last_intent"),
            model_label=meta.get("model_label") or current_model_label(),
            preview=preview,
            message_count=human_count,
        )
        imported_events = get_persistence().import_audit_events(
            payload.get("audit") or [],
            thread_id=new_id,
        )
    except Exception as exc:
        render_error(f"写入导入会话失败: {exc}")
        return None
    print()
    print(
        f"  {C.GREEN}✓ 已导入新会话:{C.RESET} "
        f"{C.CYAN}{new_id[:8]}{C.RESET}  "
        f"{C.DIM}{len(messages)} messages · {imported_events} audit events{C.RESET}"
    )
    print(f"  {C.DIM}已切换到导入的会话。{C.RESET}\n")
    return new_id


def _resolve_thread_or_current(arg: str | None, current: str) -> ThreadMeta | None:
    token = (arg or "current").strip()
    if token.lower() in {"", "current", "this", "."}:
        meta = get_persistence().get_meta(current)
        if meta is not None:
            return meta
        return ThreadMeta(
            thread_id=current,
            title=None,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
    return _resolve_thread(token)


def _split_args(args: str) -> list[str] | None:
    try:
        return shlex.split(args)
    except ValueError as exc:
        render_error(f"参数解析失败: {exc}")
        return None


def _parse_nonnegative_int(raw: str) -> int | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _print_audit_events(events: list[AuditEvent], target: ThreadMeta) -> None:
    _console.print()
    label = target.title or target.preview or target.thread_id[:8]
    _console.print(
        f"  [bold]Audit[/]  [subtle]{_truncate(label, 48)}[/]"
    )
    if not events:
        _console.print("  [subtle]（暂无审计事件）[/]")
        _console.print()
        return

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="subtle",
        border_style="muted",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("time", style="subtle", no_wrap=True, width=11)
    table.add_column("kind", style="info", no_wrap=True, width=13)
    table.add_column("target", style="subtle", no_wrap=True, width=18, overflow="ellipsis")
    table.add_column("tokens", style="subtle", justify="right", no_wrap=True, width=16)
    table.add_column("detail", no_wrap=True, overflow="ellipsis", ratio=1)

    for event in events:
        target_text = event.tool_name or event.model_label or ""
        if event.input_tokens is not None or event.output_tokens is not None:
            tokens = f"in={event.input_tokens or 0} out={event.output_tokens or 0}"
        else:
            tokens = ""
        if event.error:
            detail = f"[danger]error:[/] {event.error}"
        else:
            detail = event.args_summary or ""
        table.add_row(
            _format_ts(event.ts),
            event.kind,
            target_text,
            tokens,
            detail,
        )
    _console.print(table)
    _console.print()


def _print_usage(summary: dict, *, days: int, thread: ThreadMeta | None) -> None:
    _console.print()
    scope = f"thread {thread.thread_id[:8]}" if thread else "all threads"
    window = "all time" if days == 0 else f"{days}d"
    _console.print(f"  [bold]Usage[/]  [subtle]{scope} · {window}[/]")

    models = summary.get("models") or []
    if models:
        m_table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="subtle",
            border_style="muted",
            padding=(0, 1),
            expand=False,
            title="Models",
            title_style="subtle",
            title_justify="left",
        )
        m_table.add_column("model", style="info", no_wrap=True, max_width=28, overflow="ellipsis")
        m_table.add_column("calls", justify="right", style="subtle", no_wrap=True)
        m_table.add_column("in", justify="right", style="subtle", no_wrap=True)
        m_table.add_column("out", justify="right", style="subtle", no_wrap=True)
        m_table.add_column("cost", justify="right", no_wrap=True)
        for row in models:
            cost = estimate_cost_usd(
                row.get("model_label"),
                row.get("input_tokens"),
                row.get("output_tokens"),
            )
            m_table.add_row(
                row.get("model_label") or "unknown",
                f"{row.get('calls', 0)}",
                f"{row.get('input_tokens', 0)}",
                f"{row.get('output_tokens', 0)}",
                format_cost(cost),
            )
        _console.print(m_table)
    else:
        _console.print("  [subtle]Models: no LLM usage recorded[/]")

    tools = summary.get("tools") or []
    if tools:
        t_table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="subtle",
            border_style="muted",
            padding=(0, 1),
            expand=False,
            title="Tools / events",
            title_style="subtle",
            title_justify="left",
        )
        t_table.add_column("name", style="info", no_wrap=True, max_width=28, overflow="ellipsis")
        t_table.add_column("calls", justify="right", style="subtle", no_wrap=True)
        t_table.add_column("chars", justify="right", style="subtle", no_wrap=True)
        t_table.add_column("errors", justify="right", no_wrap=True)
        for row in tools:
            err = row.get("errors", 0)
            err_text = f"[danger]{err}[/]" if err else "[subtle]0[/]"
            t_table.add_row(
                row.get("name") or "unknown",
                f"{row.get('calls', 0)}",
                f"{row.get('result_size', 0)}",
                err_text,
            )
        _console.print(t_table)
    _console.print()


def _parse_export_args(args: str, *, current: str) -> tuple[ThreadMeta, str, Path | None] | None:
    parts = _split_args(args)
    if parts is None:
        return None
    target_arg = None
    fmt = "md"
    out_path = None
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "--format" and i + 1 < len(parts):
            fmt = parts[i + 1].lower()
            i += 2
        elif part == "--out" and i + 1 < len(parts):
            out_path = Path(parts[i + 1]).expanduser()
            i += 2
        elif part.startswith("--"):
            print(f"\n  {C.RED}未知参数：{C.RESET}{part}\n")
            return None
        elif target_arg is None:
            target_arg = part
            i += 1
        else:
            print(f"\n  {C.RED}多余参数：{C.RESET}{part}\n")
            return None
    if fmt not in {"md", "json"}:
        print(f"\n  {C.RED}--format 只能是 md 或 json{C.RESET}\n")
        return None
    target = _resolve_thread_or_current(target_arg, current)
    if target is None:
        print(f"\n  {C.RED}找不到匹配的会话：{C.RESET}{target_arg}\n")
        return None
    return target, fmt, out_path


def _thread_export_payload(
    meta: ThreadMeta,
    values: dict,
    messages: list[BaseMessage],
    events: list[AuditEvent],
) -> dict:
    state_values = {
        key: value
        for key, value in values.items()
        if key not in {"messages", "pending_shell"}
    }
    return {
        "version": 1,
        "thread_id": meta.thread_id,
        "exported_at": int(time.time()),
        "meta": asdict(meta),
        "values": state_values,
        "messages": messages_to_dict(messages),
        "audit": [asdict(event) for event in events],
    }


def _thread_export_markdown(
    meta: ThreadMeta,
    messages: list[BaseMessage],
    events: list[AuditEvent],
) -> str:
    title = meta.title or meta.preview or meta.thread_id
    lines = [
        f"# {title}",
        "",
        f"- Thread: `{meta.thread_id}`",
        f"- Exported: {_format_ts(int(time.time()))}",
        f"- Messages: {len(messages)}",
        "",
        "## Conversation",
        "",
    ]
    for message in messages:
        role = getattr(message, "type", type(message).__name__)
        name = getattr(message, "name", None)
        header = f"### {role}" + (f" · {name}" if name else "")
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        if role == "tool" and len(content) > 1200:
            content = content[:1200] + "\n\n...(tool output truncated in markdown export)"
        lines.extend([header, "", content or "_(empty)_", ""])
    if events:
        lines.extend(["## Audit Summary", ""])
        for event in events[:50]:
            detail = event.tool_name or event.model_label or event.args_summary or ""
            lines.append(
                f"- `{_format_ts(event.ts)}` `{event.kind}` "
                f"{_truncate(detail, 90)}"
            )
        lines.append("")
    return "\n".join(lines)


def _latest_human_preview(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip().replace("\n", " ")[:80] or None
    return None


def interactive_loop(app) -> int:
    """REPL 主循环：每轮读一行用户输入，按前缀决定走哪个分支。"""
    # 一个会话用一个 thread_id；用单元素列表包一层，方便 status 回调随时取最新值
    thread_box: list[str] = [str(uuid.uuid4())]
    session = make_session(_build_status_provider(thread_box))

    welcome_box()
    tips_block()

    while True:
        _draw_top_border()
        try:
            text = read_line(session)
        finally:
            # 输入结束后立刻把下边框补完，无论是正常提交还是 Ctrl-C
            _draw_bottom_border()

        if text is None:
            # Ctrl-D 或 2 秒内连按 Ctrl-C → 退出
            print(f"\n{C.DIM}再见。{C.RESET}")
            return 0
        text = text.strip()
        if not text:
            # 单次 Ctrl-C 或空输入：提示一下再继续
            print(f"  {C.DIM}(已取消；再次 Ctrl-C 退出，或输入 /exit){C.RESET}")
            continue

        # 斜杠命令：走 handle_command 分发
        if text.startswith("/"):
            keep_going, new_id = handle_command(
                text, thread_id=thread_box[0], app=app,
            )
            thread_box[0] = new_id  # /clear、/resume、/delete 可能换 id
            if not keep_going:
                return 0
            continue

        # ! 前缀：直执行 shell
        if text.startswith("!"):
            run_bang_command(text[1:])
            continue

        # 普通输入：交给 LangGraph 处理（stream_query 自己负责答案渲染）
        try:
            stream_query(app, text, thread_box[0])
        except Exception as exc:
            render_error(str(exc))
            continue


def build_parser() -> argparse.ArgumentParser:
    """命令行参数解析器：支持单次问答、--graph 导出图。"""
    parser = argparse.ArgumentParser(description="AskAnswer 命令行工具")
    parser.add_argument(
        "--graph",
        nargs="?",
        const="-",
        metavar="PATH",
        help="生成 LangGraph Mermaid 图；不填 PATH 时输出到终端",
    )
    parser.add_argument("question", nargs="?", help="要提问的内容")
    return parser


def export_graph(target: str) -> int:
    """导出 LangGraph Mermaid 图到指定路径或 stdout。"""
    try:
        mermaid = draw_search_assistant_mermaid()
    except Exception as exc:
        render_error(f"生成图失败: {exc}")
        return 1

    # "-" 表示输出到 stdout
    if target == "-":
        print(mermaid)
        return 0

    # 否则写入文件，目录不存在就先创建
    path = Path(target)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mermaid, encoding="utf-8")
    except Exception as exc:
        render_error(f"写入图文件失败: {exc}")
        return 1

    print(f"\n  {C.GREEN}✓ LangGraph 图已生成:{C.RESET} {path}\n")
    return 0


def main() -> int:
    """CLI 程序入口：解析参数 → 注册 atexit → 进入对应模式。"""
    parser = build_parser()
    args = parser.parse_args()

    # 保证退出时关闭 MCP 后台 loop，避免守护线程残留
    atexit.register(_mcp_shutdown)
    # 关闭 SQLite 连接 —— atexit 是 LIFO，shutdown_persistence 会先于 _mcp_shutdown 调用，
    # 两者无依赖关系，顺序无所谓。
    atexit.register(shutdown_persistence)

    # 仅导图模式：不需要构建主图，直接输出 Mermaid（draw_search_assistant_mermaid
    # 内部走 InMemorySaver，不会触发 ~/.askanswer/state.db 创建）
    if args.graph is not None:
        return export_graph(args.graph)

    try:
        # seed 一次注册表（内置 + sql + 任何已存在的 MCP），后续节点直接用
        get_registry()
        app = create_search_assistant()
    except Exception as exc:
        render_error(f"初始化失败: {exc}")
        return 1

    # 单次问答模式：命令行直接给了 question
    if args.question:
        thread_id = str(uuid.uuid4())
        try:
            stream_query(app, args.question, thread_id)
        except Exception as exc:
            render_error(str(exc))
            return 1
        return 0

    # 没问题参数则进入 REPL
    return interactive_loop(app)


if __name__ == "__main__":
    raise SystemExit(main())
