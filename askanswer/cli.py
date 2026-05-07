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
import os
import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding

from .graph import create_search_assistant, draw_search_assistant_mermaid
from .intents import get_intent_registry
from .load import current_model_label, model, set_model
from .mcp import get_manager as _mcp_manager, shutdown_manager as _mcp_shutdown
from .persistence import (
    ThreadMeta,
    get_persistence,
    shutdown_persistence,
)
from .registry import get_registry
from .schema import ContextSchema
from .tools import check_dangerous, execute_shell_command, gen_shell_command_spec
from .ui_input import SLASH_COMMANDS, cmd_meta, make_session, read_line
from .ui_select import CANCELLED, select_option
from .ui_spinner import Spinner


# rich 控制台：仅用于把最终答案以 Markdown 形式渲染
_console = Console()


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
        f"{C.DIM}model: {_current_model_name()}{C.RESET}",
    ]
    print()
    print(f"{C.ORANGE}╭{border}╮{C.RESET}")
    for line in lines:
        # 用 _pad 对齐右侧边框，避免 CJK/ANSI 影响列对齐
        print(f"{C.ORANGE}│{C.RESET} {_pad(line, inner)} {C.ORANGE}│{C.RESET}")
    print(f"{C.ORANGE}╰{border}╯{C.RESET}")


def tips_block() -> None:
    """欢迎面板下方的“使用小贴士”。"""
    print()
    print(f" {C.BOLD}Tips for getting started:{C.RESET}")
    print()
    print(f" {C.DIM}1.{C.RESET} 提出任何问题，我会进行搜索并整理答案")
    print(f" {C.DIM}2.{C.RESET} 问题越具体，结果越精准")
    print(f" {C.DIM}3.{C.RESET} 输入 {C.CYAN}/help{C.RESET} 查看所有命令")
    print()


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
    print()
    print(f" {C.BOLD}Status{C.RESET}")
    print(f"   {C.DIM}thread:{C.RESET}  {thread_id}")
    # 当前线程标题（若已命名）
    try:
        meta = get_persistence().get_meta(thread_id)
    except Exception:
        meta = None
    if meta and meta.title:
        print(f"   {C.DIM}title:{C.RESET}   {meta.title}")
    print(f"   {C.DIM}cwd:{C.RESET}     {Path.cwd()}")
    print(f"   {C.DIM}model:{C.RESET}   {current_model_label()}")
    # 持久化信息：DB 路径 + 已存的线程总数
    try:
        pm = get_persistence()
        thread_count = len(pm.list_threads(limit=10000))
        print(f"   {C.DIM}store:{C.RESET}   {pm.db_path}  {C.DIM}({thread_count} threads){C.RESET}")
    except Exception:
        pass
    servers = _mcp_manager().list_servers()
    if servers:
        # 简洁展示：name(工具数量)，多个 server 用逗号分隔
        summary = ", ".join(f"{s['name']}({s['tools']})" for s in servers)
        print(f"   {C.DIM}mcp:{C.RESET}     {summary}")
    else:
        print(f"   {C.DIM}mcp:{C.RESET}     {C.DIM}（未连接，/mcp <url> 添加）{C.RESET}")
    print()


# ── Streaming progress ────────────────────────────────────────────

def _marker(title: str, detail: str = "", elapsed: float | None = None) -> str:
    """生成一行节点进度标记：⏺ Title(detail)  (1.2s)。"""
    body = f"{C.BOLD}{title}{C.RESET}"
    if detail:
        body += f"{C.DIM}({detail}){C.RESET}"
    if elapsed is not None and elapsed >= 0.05:
        # ≥50ms 才打耗时，纯条件判断节点（瞬时）就不显示了
        body += f"  {C.DIM}({elapsed:.1f}s){C.RESET}"
    return f"  {C.ORANGE}⏺{C.RESET} {body}"


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

    # 兜底：若节点流里没拿到 final_answer，从 state 里找最后一条消息内容
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

    # Live 已经把答案画在屏上了，就只补一个空行做间距；
    # 没走 Live 的（典型场景：模型不支持 stream，或 sorcery 重写答案）才走传统渲染。
    if live_state.get("streamed"):
        print()
    else:
        render_answer(final_answer or "未生成答案。")

    # 持久化线程元数据：每次问答后写一行（首次写入会自动取 preview 前 30 字符做 title）。
    # 失败不影响主流程 —— 用户拿到回答比记账更重要。
    try:
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
        # 第一次拿到 user-facing token：让 spinner 让位，启动 Live 渲染
        spinner.stop()
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
    print()
    _console.print(Padding(Markdown(answer or "_(空答案)_"), (0, 2)))
    print()


def render_error(message: str) -> None:
    """统一的错误样式：红色叉号 + 灰色细节。"""
    print()
    print(f"  {C.RED}✗ 运行失败{C.RESET}")
    print(f"  {C.DIM}{message}{C.RESET}")
    print()


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
        handle_model_command(tail)
    elif head_lc == "/mcp":
        handle_mcp_command(tail)
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
    else:
        print(
            f"\n  {C.RED}未知命令：{C.RESET}{stripped}  "
            f"({C.CYAN}/help{C.RESET} 查看可用命令)\n"
        )
    return True, thread_id


def handle_model_command(args: str) -> None:
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

    print()
    print(f"  {C.GREEN}✓ 已切换模型:{C.RESET} {C.BOLD}{label}{C.RESET}")
    print()


def handle_mcp_command(args: str) -> None:
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
        _add_mcp_url(first, rest or None)
    else:
        print(
            f"\n  {C.RED}无法识别 /mcp 参数：{C.RESET}{args}\n"
            f"  {C.DIM}提示：URL 需以 http:// 或 https:// 开头{C.RESET}"
        )
        mcp_help_block()


def _add_mcp_url(url: str, name: str | None) -> None:
    """连接一个 HTTP/SSE 类的 MCP 服务，并刷新工具注册表。"""
    try:
        registered = _mcp_manager().add_url(url, name=name)
    except Exception as exc:
        render_error(f"MCP 连接失败: {exc}")
        return
    # 注册表里 mcp:* 这一片需要重新拉取
    get_registry().refresh_mcp()
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
    """打印某个 server（或全部 server）下的工具列表。"""
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

    print()
    if not threads:
        hint = f"无匹配 '{keyword}'" if keyword else "暂无历史会话（先聊几句吧）"
        print(f"  {C.DIM}（{hint}）{C.RESET}")
        print()
        return

    title_word = f" · 关键词: {keyword}" if keyword else ""
    print(f"  {C.BOLD}Threads ({len(threads)}){title_word}{C.RESET}")
    for i, m in enumerate(threads, 1):
        marker = f"{C.GREEN}●{C.RESET}" if m.thread_id == current else " "
        # 选 title 优先，其次 preview，否则给个占位
        text = (m.title or m.preview or "(空)").strip().replace("\n", " ")
        text = _truncate(text, 50)
        intent = (m.last_intent or "—")[:8]
        print(
            f"  {marker} {C.DIM}{i:>2}.{C.RESET} "
            f"{C.CYAN}{m.thread_id[:8]}{C.RESET}  "
            f"{C.DIM}{_format_ts(m.updated_at)}{C.RESET}  "
            f"{C.DIM}{intent:<8}{C.RESET} "
            f"{C.DIM}{m.message_count}msg{C.RESET}  "
            f"{text}"
        )
    print()
    print(
        f"  {C.DIM}用法：{C.RESET}"
        f"{C.CYAN}/resume <序号|id>{C.RESET}{C.DIM} 恢复 · {C.RESET}"
        f"{C.CYAN}/title <名字>{C.RESET}{C.DIM} 命名当前 · {C.RESET}"
        f"{C.CYAN}/delete <序号|id>{C.RESET}{C.DIM} 删除{C.RESET}"
    )
    print()


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
    print(f"    {C.DIM}操作不可撤销（同步清除 checkpoints + thread_meta）。{C.RESET}")
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
