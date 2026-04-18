from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    generate_answer_node,
    sorcery_answer_node,
    tavily_search_node,
    tools_node,
    understand_query_node,
)
from .state import SearchState


def route_from_answer(state: SearchState):
    if state["step"] == "tool_called":
        return "tools"
    return "sorcery"


def route_from_sorcery(state: SearchState):
    if state["step"] == "retry_search":
        return "search"
    return END


def create_search_assistant():
    workflow = StateGraph(SearchState)

    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    workflow.add_node("sorcery", sorcery_answer_node)
    workflow.add_node("tools", tools_node)

    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_conditional_edges(
        "answer",
        route_from_answer,
        {"tools": "tools", "sorcery": "sorcery"},
    )
    workflow.add_edge("tools", "answer")
    workflow.add_conditional_edges(
        "sorcery",
        route_from_sorcery,
        {"search": "search", END: END},
    )

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app
