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


def help_block() -> None:
    """/help 输出的命令清单。"""
    print()
    print(f" {C.BOLD}Commands{C.RESET}")
    print(f"   {C.CYAN}/help{C.RESET}     显示此帮助")
    print(f"   {C.CYAN}/clear{C.RESET}    清屏并开始新会话（旧会话保留，可 /threads 找回）")
    print(f"   {C.CYAN}/status{C.RESET}   查看当前会话信息")
    print(f"   {C.CYAN}/model{C.RESET}    查看或切换模型 ({C.DIM}/model <provider:name>{C.RESET})")
    print(f"   {C.CYAN}/mcp{C.RESET}      管理 MCP 服务 ({C.DIM}/mcp 查看子命令{C.RESET})")
    print(f"   {C.CYAN}/threads{C.RESET}  列出历史会话 ({C.DIM}/threads [关键词]{C.RESET})")
    print(f"   {C.CYAN}/resume{C.RESET}   恢复指定会话 ({C.DIM}/resume <序号|id 前缀>{C.RESET})")
    print(f"   {C.CYAN}/title{C.RESET}    给当前会话命名 ({C.DIM}/title <名字>{C.RESET})")
    print(f"   {C.CYAN}/delete{C.RESET}   删除会话 ({C.DIM}/delete <序号|id 前缀>{C.RESET})")
    print(f"   {C.CYAN}/exit{C.RESET}     退出 (也可 /quit, /q, Ctrl-D)")
    print(f"   {C.CYAN}!<cmd>{C.RESET}    直接执行 shell 命令 (如 !ls -la)")
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


def _render_node_update(node: str, update: dict, final_answer: str) -> str:
    """根据节点名渲染对应的进度标记，并把 final_answer 顺手记下来。"""
    if node == "understand":
        detail = _truncate(_intent_cli_label(update))
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
            directive = update.get("retry_directive") or {}
            nsq = _truncate(update.get("search_query", "") or directive.get("instruction", ""))
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
        help_block()
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
            keep_going, thread_id = handle_command(text, thread_id=thread_id, app=app)
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
