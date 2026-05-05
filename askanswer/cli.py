# 命令行入口与交互式 REPL。
# 设计要点：
# - 用 ANSI 转义自绘“面板/边框”样式，无第三方 TUI 依赖；
# - app.stream(stream_mode="updates") 拿到的每个节点更新都渲染成一行进度标记；
# - HITL（人机确认）场景下监听 __interrupt__，从 CLI 拿用户输入并 Command(resume=...) 续跑；
# - 斜杠命令（/help、/clear、/status、/model、/mcp、/exit）由 handle_command 路由；
# - `!<cmd>` 前缀绕开 LangGraph 直接 shell 执行（带危险检查）。
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

from .graph import create_search_assistant, draw_search_assistant_mermaid
from .load import current_model_label, model, set_model
from .mcp import get_manager as _mcp_manager, shutdown_manager as _mcp_shutdown
from .registry import get_registry
from .schema import ContextSchema
from .tools import check_dangerous, execute_shell_command, gen_shell_command_spec


# rich 控制台：仅用于把最终答案以 Markdown 形式渲染
_console = Console()


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


def help_block() -> None:
    """/help 输出的命令清单。"""
    print()
    print(f" {C.BOLD}Commands{C.RESET}")
    print(f"   {C.CYAN}/help{C.RESET}    显示此帮助")
    print(f"   {C.CYAN}/clear{C.RESET}   清屏并开始新会话")
    print(f"   {C.CYAN}/status{C.RESET}  查看当前会话信息")
    print(f"   {C.CYAN}/model{C.RESET}   查看或切换模型 ({C.DIM}/model <provider:name>{C.RESET})")
    print(f"   {C.CYAN}/mcp{C.RESET}     管理 MCP 服务 ({C.DIM}/mcp 查看子命令{C.RESET})")
    print(f"   {C.CYAN}/exit{C.RESET}    退出 (也可 /quit, /q, Ctrl-D)")
    print(f"   {C.CYAN}!<cmd>{C.RESET}   直接执行 shell 命令 (如 !ls -la)")
    print()


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
    """/status 输出：当前线程 ID、CWD、模型、MCP 连接状态。"""
    print()
    print(f" {C.BOLD}Status{C.RESET}")
    print(f"   {C.DIM}thread:{C.RESET}  {thread_id}")
    print(f"   {C.DIM}cwd:{C.RESET}     {Path.cwd()}")
    print(f"   {C.DIM}model:{C.RESET}   {current_model_label()}")
    servers = _mcp_manager().list_servers()
    if servers:
        # 简洁展示：name(工具数量)，多个 server 用逗号分隔
        summary = ", ".join(f"{s['name']}({s['tools']})" for s in servers)
        print(f"   {C.DIM}mcp:{C.RESET}     {summary}")
    else:
        print(f"   {C.DIM}mcp:{C.RESET}     {C.DIM}（未连接，/mcp <url> 添加）{C.RESET}")
    print()


# ── Streaming progress ────────────────────────────────────────────

def _marker(title: str, detail: str = "") -> str:
    """生成一行节点进度标记：⏺ Title(detail)。"""
    body = f"{C.BOLD}{title}{C.RESET}"
    if detail:
        body += f"{C.DIM}({detail}){C.RESET}"
    return f"  {C.ORANGE}⏺{C.RESET} {body}"


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
    """跑一次完整的图调用，期间逐节点渲染进度，并处理 HITL interrupt。"""
    final_answer = ""
    # 同一个 thread_id 表示同一段会话，复用 checkpointer 的状态
    config = {"configurable": {"thread_id": thread_id}}
    context = runtime_context or _runtime_context()
    # 第一次进入图：把用户消息塞进 messages；resume 后会被替换成 Command(...)
    graph_input: object = {"messages": [HumanMessage(content=query)]}

    print()
    while True:
        interrupt_payload = None
        # stream_mode="updates"：每个节点产生增量时回调；__interrupt__ 是特殊通道
        for chunk in app.stream(
            graph_input,
            config=config,
            context=context,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    # 节点抛出 interrupt() 时 LangGraph 通过这个伪 node 通知我们
                    interrupt_payload = _extract_interrupt_value(update)
                    continue
                if not isinstance(update, dict):
                    continue
                final_answer = _render_node_update(node, update, final_answer)

        # 有些 langgraph 版本不会通过 updates 通道暴露 __interrupt__，
        # 流结束后再从 state 探一遍挂起任务上的 interrupt，作为兜底。
        if interrupt_payload is None:
            interrupt_payload = _pending_interrupt(app, config)

        if interrupt_payload is None:
            break

        # 拿到中断后弹出确认 UI；用户的回应作为 Command(resume=...) 喂回去继续跑
        resume_value = _prompt_shell_confirmation(interrupt_payload)
        graph_input = Command(resume=resume_value)

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

    return final_answer or "未生成答案。"


def _render_node_update(node: str, update: dict, final_answer: str) -> str:
    """根据节点名渲染对应的进度标记，并把 final_answer 顺手记下来。"""
    if node == "understand":
        # 渲染 intent 与对应的关键信息（搜索词 / 文件路径）
        intent = update.get("intent", "")
        if intent == "file_read":
            detail = f"file_read: {_truncate(update.get('file_path', ''))}"
        elif intent == "sql":
            detail = "sql"
        elif intent == "chat":
            detail = "chat"
        else:
            detail = f"search: {_truncate(update.get('search_query', ''))}"
        print(_marker("Understand", detail))
    elif node == "file_read":
        # （兼容老拓扑）file_read 节点已合并到 react 的 read_file 工具
        if update.get("final_answer"):
            final_answer = update["final_answer"]
        print(_marker("FileRead", "读取完成"))
    elif node == "search":
        # （兼容老拓扑）search 作为独立节点的版本
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
        # 兜底：未知节点也给一行标记，至少能看到流转
        print(_marker(node))
    return final_answer


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
    """用文本 UI 提示用户确认 shell 命令；支持 y/n/e（编辑后再生成）。"""
    data = payload if isinstance(payload, dict) else {}
    command = data.get("command") or str(payload)
    explanation = data.get("explanation") or ""
    instruction = data.get("instruction") or ""

    # 循环：用户选 e 编辑指令时，重新生成命令并再次询问
    while True:
        print()
        print(f"  {C.ORANGE}⏸{C.RESET}  {C.BOLD}需要确认 Shell 命令{C.RESET}")
        print(f"    {C.DIM}命令：{C.RESET}{C.CYAN}{command}{C.RESET}")
        if explanation:
            print(f"    {C.DIM}说明：{C.RESET}{explanation}")
        print(
            f"    {C.DIM}选项：{C.RESET}"
            f"{C.GREEN}y{C.RESET}=执行  "
            f"{C.RED}n{C.RESET}=取消  "
            f"{C.GOLD}e{C.RESET}=补充说明后重新生成"
        )
        try:
            reply = input(f"    {C.ORANGE}你的选择 (y/N/e):{C.RESET} ").strip().lower()
        except EOFError:
            # Ctrl-D：等价于取消
            reply = ""
        except KeyboardInterrupt:
            # Ctrl-C：等价于取消
            reply = ""
            print()

        if reply in ("y", "yes"):
            # 批准：让节点真正执行命令
            return {"approve": True, "command": command, "explanation": explanation}
        if reply in ("e", "edit", "more", "add"):
            # 让用户输入补充说明，再让 LLM 重新生成命令
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
        # 其它输入（包括 n/no/空）一律视为取消
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

def _prompt_boxed_input() -> str:
    """绘制带边框的输入提示框，用户在框内敲一行；回车后再画底边框。"""
    w = _term_width()
    border = "─" * (w - 2)
    print(f"{C.ORANGE}╭{border}╮{C.RESET}")
    sys.stdout.write(f"{C.ORANGE}│{C.RESET} {C.ORANGE}>{C.RESET} ")
    sys.stdout.flush()
    try:
        text = input()
    finally:
        # finally 保证即使输入异常也会把底边框补完，视觉对齐不破
        print(f"{C.ORANGE}╰{border}╯{C.RESET}")
    return text


def handle_command(cmd: str, *, thread_id: str) -> tuple[bool, str]:
    """斜杠命令路由。返回 (是否继续运行, 当前 thread_id)。"""
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
        # 清屏 + 新建一个 thread_id：等价于一段全新对话
        os.system("cls" if os.name == "nt" else "clear")
        welcome_box()
        thread_id = str(uuid.uuid4())
        print(f"\n  {C.DIM}已开始新会话：{thread_id[:8]}…{C.RESET}\n")
    elif head_lc == "/status":
        status_block(thread_id)
    elif head_lc == "/model":
        handle_model_command(tail)
    elif head_lc == "/mcp":
        handle_mcp_command(tail)
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


def interactive_loop(app) -> int:
    """REPL 主循环：每轮读一行用户输入，按前缀决定走哪个分支。"""
    # 一个会话用一个 thread_id；/clear 会替换它
    thread_id = str(uuid.uuid4())

    welcome_box()
    tips_block()

    while True:
        try:
            text = _prompt_boxed_input().strip()
        except EOFError:
            # Ctrl-D：退出
            print(f"\n{C.DIM}再见。{C.RESET}")
            return 0
        except KeyboardInterrupt:
            # Ctrl-C：仅取消当前输入，回到下一轮
            print(f"\n{C.DIM}(已取消，输入 /exit 退出){C.RESET}")
            continue

        if not text:
            continue

        # 斜杠命令：走 handle_command 分发
        if text.startswith("/"):
            keep_going, thread_id = handle_command(text, thread_id=thread_id)
            if not keep_going:
                return 0
            continue

        # ! 前缀：直执行 shell
        if text.startswith("!"):
            run_bang_command(text[1:])
            continue

        # 普通输入：交给 LangGraph 处理
        try:
            answer = stream_query(app, text, thread_id)
        except Exception as exc:
            render_error(str(exc))
            continue

        render_answer(answer)


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

    # 仅导图模式：不需要构建主图，直接输出 Mermaid
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
            answer = stream_query(app, args.question, thread_id)
        except Exception as exc:
            render_error(str(exc))
            return 1
        render_answer(answer)
        return 0

    # 没问题参数则进入 REPL
    return interactive_loop(app)


if __name__ == "__main__":
    raise SystemExit(main())
