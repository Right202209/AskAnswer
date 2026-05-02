from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import sorcery_answer_node, understand_query_node
from .react import build_react_subgraph
from .schema import ContextSchema
from .state import SearchState


def route_from_sorcery(state: SearchState):
    if state["step"] == "retry_search":
        return "answer"
    return END


def create_search_assistant():
    workflow = StateGraph(SearchState, context_schema=ContextSchema)
    react = build_react_subgraph()

    workflow.add_node("understand", understand_query_node)
    workflow.add_node("answer", react)
    workflow.add_node("sorcery", sorcery_answer_node)

    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "answer")
    workflow.add_edge("answer", "sorcery")
    workflow.add_conditional_edges(
        "sorcery",
        route_from_sorcery,
        {"answer": "answer", END: END},
    )

    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


def draw_search_assistant_mermaid() -> str:
    app = create_search_assistant()
    return app.get_graph().draw_mermaid()
