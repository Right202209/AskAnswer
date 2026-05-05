# React 子图：answer ⇄ tools 的循环，包含 HITL（人机确认）的 shell_plan 旁路。
# 该子图作为父图的 "answer" 节点被复用，意图无关 —— 真正的差异化体现在
# _answer_node 的 system prompt 与 tool 绑定上。
from langgraph.graph import END, START, StateGraph

from ._react_internals import (
    _answer_node,
    _route_from_answer,
    _shell_plan_node,
    _tools_node,
)
from .schema import ContextSchema
from .state import SearchState


def build_react_subgraph():
    """编译 answer ⇄ tools / shell_plan 这条 ReAct 循环，作为子图返回。

    与父图共享 ``SearchState``，因此可以直接通过 ``add_node`` 嵌入。注意 **不**
    传入 checkpointer ——子图按 per-invocation 模式继承父图的 checkpointer，
    这样 shell HITL 流程里 ``interrupt()`` 抛出的中断才能透传到父级 stream。
    """
    builder = StateGraph(SearchState, context_schema=ContextSchema)
    # answer 节点：调用 LLM、根据 intent 绑定不同工具集、产出 AIMessage 或 tool_calls
    builder.add_node("answer", _answer_node)
    # shell_plan 节点：在执行需要确认的 shell 工具前预先生成命令并写入 pending_shell
    builder.add_node("shell_plan", _shell_plan_node)
    # tools 节点：分发普通工具调用与需要确认的工具调用
    builder.add_node("tools", _tools_node)

    builder.add_edge(START, "answer")
    # answer 之后的条件分支：无 tool 调用直接结束；普通工具走 tools；需要确认的走 shell_plan
    builder.add_conditional_edges(
        "answer",
        _route_from_answer,
        {"shell_plan": "shell_plan", "tools": "tools", END: END},
    )
    # 规划完命令后必然进入 tools 执行（其内部可能 interrupt 等待用户确认）
    builder.add_edge("shell_plan", "tools")
    # 工具执行完总是回到 answer，由 LLM 决定继续调用工具还是输出最终回答
    builder.add_edge("tools", "answer")
    return builder.compile()
