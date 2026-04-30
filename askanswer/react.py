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
    """Compile the answer ⇄ tools / shell_plan ReAct loop as a subgraph.

    Shares ``SearchState`` with the parent graph so it can be added directly via
    ``add_node``. No checkpointer is passed — the subgraph inherits the parent's
    checkpointer (per-invocation mode), which is what lets ``interrupt()`` from
    the shell HITL flow propagate to the parent stream.
    """
    builder = StateGraph(SearchState, context_schema=ContextSchema)
    builder.add_node("answer", _answer_node)
    builder.add_node("shell_plan", _shell_plan_node)
    builder.add_node("tools", _tools_node)

    builder.add_edge(START, "answer")
    builder.add_conditional_edges(
        "answer",
        _route_from_answer,
        {"shell_plan": "shell_plan", "tools": "tools", END: END},
    )
    builder.add_edge("shell_plan", "tools")
    builder.add_edge("tools", "answer")
    return builder.compile()
