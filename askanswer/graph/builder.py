"""主图组装：单层扁平 7 节点（架构文档 4.2）。

    START → route → clarify → reason ─┬─(无工具)────────→ review ─┬→ END
                                      ├─(普通工具)→ act ──→ reason │
                                      └─(需审批)→ plan_approvals   └→ reason（重试）
                                           → ask_approval ⟲ → act

不再有「父图 + react 子图」：interrupt 不用跨子图透传；审批拆成三个节点，结构上
消除恢复时重复执行（P9）；每个节点都由 ``make_xxx_node(container)`` 生成，可单测。
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from askanswer.graph.context import ContextSchema
from askanswer.graph.edges import (
    make_route_after_reason,
    route_after_approval,
    route_after_review,
)
from askanswer.graph.nodes.act import make_act_node
from askanswer.graph.nodes.approvals import make_ask_approval_node, make_plan_approvals_node
from askanswer.graph.nodes.clarify import make_clarify_node
from askanswer.graph.nodes.reason import make_reason_node
from askanswer.graph.nodes.review import make_review_node
from askanswer.graph.nodes.route import make_route_node
from askanswer.graph.state import SearchState
from askanswer.runtime import Container, build_container

NODES = ("route", "clarify", "reason", "plan_approvals", "ask_approval", "act", "review")


def build_graph(container: Container) -> StateGraph:
    """按 container 组装（未编译的）主图。"""
    c = container
    g = StateGraph(SearchState, context_schema=ContextSchema)
    g.add_node("route", make_route_node(c))
    g.add_node("clarify", make_clarify_node(c))
    g.add_node("reason", make_reason_node(c))
    g.add_node("plan_approvals", make_plan_approvals_node(c))
    g.add_node("ask_approval", make_ask_approval_node(c))
    g.add_node("act", make_act_node(c))
    g.add_node("review", make_review_node(c))

    g.add_edge(START, "route")
    g.add_edge("route", "clarify")
    g.add_edge("clarify", "reason")
    g.add_conditional_edges(
        "reason",
        make_route_after_reason(c),
        {"plan_approvals": "plan_approvals", "act": "act", "review": "review"},
    )
    g.add_edge("plan_approvals", "ask_approval")
    g.add_conditional_edges(
        "ask_approval", route_after_approval, {"ask_approval": "ask_approval", "act": "act"}
    )
    g.add_edge("act", "reason")
    g.add_conditional_edges("review", route_after_review, {"reason": "reason", END: END})
    return g


def create_search_assistant(
    checkpointer: BaseCheckpointSaver | None = None,
    *,
    container: Container | None = None,
):
    """构建并编译主图，返回可直接 invoke / stream 的 LangGraph 应用。

    :param checkpointer: ``None`` 时用 ``container.store.checkpointer``（SQLite 持久化）；
        想跳过持久化（``--graph`` 导出、单测）时传 ``InMemorySaver()``。
    :param container: 依赖组装根；``None`` 使用进程默认实现。单测可注入替身。
    """
    c = container or build_container()
    if checkpointer is None:
        checkpointer = c.store.checkpointer
    return build_graph(c).compile(checkpointer=checkpointer)


def draw_search_assistant_mermaid() -> str:
    """导出主图的 Mermaid 文本表示，供 ``askanswer --graph`` 使用。

    显式传 ``InMemorySaver`` 避免无谓地在 ``~/.askanswer/state.db`` 创建空 DB。
    """
    app = create_search_assistant(checkpointer=InMemorySaver())
    return app.get_graph().draw_mermaid()
