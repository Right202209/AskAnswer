# 节点进度渲染：把图节点完成事件翻译成 ``⏺ Node  detail  · 1.2s`` 标记行。
#
# 纯展示：由 stream.py 在消费 runner 的 node 事件时调用；spinner 生命周期由调用方管。
from __future__ import annotations

import re

from ..intents import get_intent_registry
from .text import _truncate
from .theme import C

# 识别搜索结果里 “1. **标题**” 这种条目的行首
_HIT_RE = re.compile(r"^\d+\.\s+\*\*", re.MULTILINE)


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
    "understand":    "理解意图…",
    "clarify":       "澄清需求…",
    "answer":        "思考中…",
    "tools":         "执行工具…",
    "confirm_plan":  "规划待确认操作…",
    "shell_plan":    "规划 shell 命令…",
    "search":        "联网搜索…",
    "file_read":     "读取文件…",
    "sorcery":       "评估答案质量…",
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
    elif node == "clarify":
        # 只有真的澄清出结果（update 非空）才打标记；空 update 在 _on_node_update 已拦截。
        print(_marker("Clarify", _clarify_detail(update), elapsed))
    elif node in ("confirm_plan", "shell_plan"):
        plans = update.get("pending_confirmations") or update.get("pending_shell") or {}
        detail = f"规划 {len(plans)} 项确认" if plans else "规划完成"
        print(_marker("Confirm", detail, elapsed))
    else:
        # 兜底：未知节点也给一行标记，至少能看到流转
        print(_marker(node, "", elapsed))
    return final_answer


def _intent_cli_label(update: dict) -> str:
    return get_intent_registry().get(update.get("intent")).cli_label(update)


def _clarify_detail(update: dict) -> str:
    """把 clarify 节点的 update（并回的字段）翻成一行人类可读的进度说明。"""
    if update.get("file_path"):
        return f"文件：{_truncate(update['file_path'], 40)}"
    if update.get("intent"):
        return "改用通用知识作答"
    if update.get("user_query"):
        return "已细化研究范围"
    return "已澄清"
