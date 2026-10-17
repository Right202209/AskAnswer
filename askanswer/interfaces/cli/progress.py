# 节点进度渲染：把图节点完成事件翻译成 ``⏺ Node  detail  · 1.2s`` 标记行。
#
# 纯展示：由 stream.py 在消费 runner 的 node 事件时调用；spinner 生命周期由调用方管。
from __future__ import annotations

from askanswer.intents import get_intent_registry
from askanswer.interfaces.cli.text import _truncate
from askanswer.interfaces.cli.theme import C

# 识别搜索结果里 “1. **标题**” 这种条目的行首


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
    "route":          "理解意图…",
    "clarify":        "澄清需求…",
    "reason":         "执行工具…",      # reason 产出 tool_calls 后下一步是执行
    "act":            "思考中…",        # 工具执行完回到 reason
    "plan_approvals": "规划待确认操作…",
    "ask_approval":   "等待确认…",
    "review":         "评估答案质量…",
}


def _phase_text(node: str) -> str:
    return _PHASE_TEXT.get(node, "思考中…")


def _render_node_update_safely(
    node: str, update: dict, final_answer: str, elapsed: float, spinner,
) -> str:
    """``spinner.freeze_for`` 不能跨返回值传递，这里包一层用列表传出。"""
    holder = [final_answer]

    def _do():
        holder[0] = _render_node_update(node, update, final_answer, elapsed)

    spinner.freeze_for(_do)
    return holder[0]


def _render_node_update(
    node: str, update: dict, final_answer: str, elapsed: float | None = None,
) -> str:
    """根据节点名渲染对应的进度标记，并把 final_answer 顺手记下来。"""
    if node == "route":
        detail = _truncate(_intent_cli_label(update))
        print(_marker("Route", detail, elapsed))
    elif node == "reason":
        # 产出最终答案的 reason 在 stream._on_node_update 里特别处理（含 Live 收尾）；
        # 这里只渲染「决定调用工具」的那一步。
        if update.get("final_answer"):
            final_answer = update["final_answer"]
            print(_marker("Answer", "完成", elapsed))
        else:
            print(_marker("Reason", _tool_calls_detail(update), elapsed))
    elif node == "review":
        if update.get("final_answer"):
            final_answer = update["final_answer"]
        if update.get("step") == "retry_search":
            directive = update.get("retry_directive") or {}
            new_query = (update.get("slots") or {}).get("search_query", "")
            nsq = _truncate(new_query or directive.get("instruction", ""))
            print(_marker("Review", f"不够好，重试：{nsq}", elapsed))
        else:
            print(_marker("Review", "通过", elapsed))
    elif node == "act":
        print(_marker("Act", "工具执行完成", elapsed))
    elif node == "clarify":
        # 只有真的澄清出结果（update 非空）才打标记；空 update 在 _on_node_update 已拦截。
        print(_marker("Clarify", _clarify_detail(update), elapsed))
    elif node == "ask_approval":
        plans = update.get("pending_confirmations") or {}
        blocked = sum(1 for p in plans.values() if isinstance(p, dict) and p.get("blocked"))
        detail = f"拦截 {blocked} 项" if blocked else "已记录决定"
        print(_marker("Approve", detail, elapsed))
    elif node == "plan_approvals":
        plans = update.get("pending_confirmations") or {}
        detail = f"规划 {len(plans)} 项确认" if plans else "规划完成"
        print(_marker("Confirm", detail, elapsed))
    else:
        # 兜底：未知节点也给一行标记，至少能看到流转
        print(_marker(node, "", elapsed))
    return final_answer


def _tool_calls_detail(update: dict) -> str:
    """reason 决定调用的工具名，如「调用 tavily_search, read_file」。"""
    names: list[str] = []
    for message in update.get("messages") or []:
        for tc in getattr(message, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else None
            if name and name not in names:
                names.append(name)
    return _truncate("调用 " + ", ".join(names)) if names else "调用工具"


def _intent_cli_label(update: dict) -> str:
    return get_intent_registry().get(update.get("intent")).cli_label(update)


def _clarify_detail(update: dict) -> str:
    """把 clarify 节点的 update（并回的字段）翻成一行人类可读的进度说明。"""
    file_path = (update.get("slots") or {}).get("file_path")
    if file_path:
        return f"文件：{_truncate(file_path, 40)}"
    if update.get("intent"):
        return "改用通用知识作答"
    if update.get("user_query"):
        return "已细化研究范围"
    return "已澄清"
