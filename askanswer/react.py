# React 子图：answer ⇄ tools 的循环，包含 HITL（人机确认）的 confirm_plan 旁路。
# 该子图作为父图的 "answer" 节点被复用，意图无关 —— 真正的差异化体现在
# _answer_node 的 system prompt 与 tool 绑定上。
from langgraph.graph import END, START, StateGraph

from ._react_internals import (
    _answer_node,
    _confirm_plan_node,
    _route_from_answer,
    _tools_node,
)
from .clarify import clarify_node
from .schema import ContextSchema
from .state import SearchState


def build_react_subgraph():
    """编译 answer ⇄ tools / confirm_plan 这条 ReAct 循环，作为子图返回。

    与父图共享 ``SearchState``，因此可以直接通过 ``add_node`` 嵌入。注意 **不**
    传入 checkpointer ——子图按 per-invocation 模式继承父图的 checkpointer，
    这样 HITL 流程里 ``interrupt()`` 抛出的中断才能透传到父级 stream。
    """
    builder = StateGraph(SearchState, context_schema=ContextSchema)
    # clarify 节点：answer 之前的通用澄清入口（缺路径/缺 DSN/范围不清时 interrupt 询问），
    # 仅首轮触发、不改父图拓扑；无需澄清时是零成本直通。
    builder.add_node("clarify", clarify_node)
    # answer 节点：调用 LLM、根据 intent 绑定不同工具集、产出 AIMessage 或 tool_calls
    builder.add_node("answer", _answer_node)
    # confirm_plan 节点：在执行需要确认的工具前按确认类规划动作并写入 pending_confirmations
    builder.add_node("confirm_plan", _confirm_plan_node)
    # tools 节点：分发普通工具调用与需要确认的工具调用
    builder.add_node("tools", _tools_node)

    # 入口先过 clarify，再进 answer
    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", "answer")
    # answer 之后的条件分支：无 tool 调用直接结束；普通工具走 tools；需要确认的走 confirm_plan
    builder.add_conditional_edges(
        "answer",
        _route_from_answer,
        {"confirm_plan": "confirm_plan", "tools": "tools", END: END},
    )
    # 规划完动作后必然进入 tools 执行（其内部可能 interrupt 等待用户确认）
    builder.add_edge("confirm_plan", "tools")
    # 工具执行完总是回到 answer，由 LLM 决定继续调用工具还是输出最终回答
    builder.add_edge("tools", "answer")
    return builder.compile()
