# React 子图：answer ⇄ tools 的循环，包含 HITL（人机确认）的 confirm_plan 旁路。
# 该子图作为父图的 "answer" 节点被复用，意图无关 —— 真正的差异化体现在
# _answer_node（answering.py）的 system prompt、模型路由与 tool 绑定上。
from langgraph.graph import END, START, StateGraph

from ._react_internals import (
    make_ask_approval_node,
    make_confirm_plan_node,
    make_route_from_answer,
    make_tools_node,
    route_after_approval,
)
from .answering import make_answer_node
from .clarify import make_clarify_node
from .runtime import Container, build_container
from .schema import ContextSchema
from .state import SearchState


def build_react_subgraph(container: Container | None = None):
    """编译 answer ⇄ tools / confirm_plan 这条 ReAct 循环，作为子图返回。

    与父图共享 ``SearchState``，因此可以直接通过 ``add_node`` 嵌入。注意 **不**
    传入 checkpointer ——子图按 per-invocation 模式继承父图的 checkpointer，
    这样 HITL 流程里 ``interrupt()`` 抛出的中断才能透传到父级 stream。
    """
    c = container or build_container()
    builder = StateGraph(SearchState, context_schema=ContextSchema)
    # clarify 节点：answer 之前的通用澄清入口（缺路径/缺 DSN/范围不清时 interrupt 询问），
    # 仅首轮触发、不改父图拓扑；无需澄清时是零成本直通。
    builder.add_node("clarify", make_clarify_node(c))
    # answer 节点：调用 LLM、根据 intent 绑定不同工具集、产出 AIMessage 或 tool_calls
    builder.add_node("answer", make_answer_node(c))
    # confirm_plan 节点：在执行需要确认的工具前按确认类规划动作并写入 pending_confirmations
    builder.add_node("confirm_plan", make_confirm_plan_node(c))
    # ask_approval 节点：一次只 interrupt 询问一个确认项，只读写 state、无副作用
    builder.add_node("ask_approval", make_ask_approval_node(c))
    # tools 节点：执行普通工具 + 按已记录的决定执行确认类工具（绝不 interrupt）
    builder.add_node("tools", make_tools_node(c))

    # 入口先过 clarify，再进 answer
    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", "answer")
    # answer 之后的条件分支：无 tool 调用直接结束；普通工具走 tools；需要确认的走 confirm_plan
    builder.add_conditional_edges(
        "answer",
        make_route_from_answer(c),
        {"confirm_plan": "confirm_plan", "tools": "tools", END: END},
    )
    # 规划完动作后逐项询问；全部有了决定才进入 tools 执行。
    # 把 interrupt 与副作用放在不同节点，恢复时重跑 ask_approval 不会重复执行任何工具。
    builder.add_edge("confirm_plan", "ask_approval")
    builder.add_conditional_edges(
        "ask_approval",
        route_after_approval,
        {"ask_approval": "ask_approval", "tools": "tools"},
    )
    # 工具执行完总是回到 answer，由 LLM 决定继续调用工具还是输出最终回答
    builder.add_edge("tools", "answer")
    return builder.compile()
