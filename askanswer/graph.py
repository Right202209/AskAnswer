from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    file_read_node,
    sorcery_answer_node,
    sql_agent_node,
    tavily_search_node,
    understand_query_node,
)
from .react import build_react_subgraph
from .schema import ContextSchema
from .state import SearchState


def route_from_understand(state: SearchState):
    intent = state.get("intent", "search")
    if intent == "file_read":
        return "file_read"
    if intent == "sql":
        return "sql"
    if intent == "chat":
        return "answer"
    return "search"


def route_from_sorcery(state: SearchState):
    if state["step"] == "retry_search":
        return "search"
    return END


def create_search_assistant():
    workflow = StateGraph(SearchState, context_schema=ContextSchema)
    react = build_react_subgraph()

    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", react)
    workflow.add_node("sorcery", sorcery_answer_node)
    workflow.add_node("file_read", file_read_node)
    workflow.add_node("sql", sql_agent_node)

    workflow.add_edge(START, "understand")
    workflow.add_conditional_edges(
        "understand",
        route_from_understand,
        {"file_read": "file_read", "sql": "sql", "search": "search", "answer": "answer"},
    )
    workflow.add_edge("search", "answer")
    workflow.add_edge("sql", END)
    workflow.add_edge("answer", "sorcery")
    workflow.add_edge("file_read", "sorcery")
    workflow.add_conditional_edges(
        "sorcery",
        route_from_sorcery,
        {"search": "search", END: END},
    )

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


def draw_search_assistant_mermaid() -> str:
    app = create_search_assistant()
    return app.get_graph().draw_mermaid()
