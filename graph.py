from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from AskAnswer.Node import understand_query_node, tavily_search_node, generate_answer_node, sorcery_answer_node, \
    tools_node
from AskAnswer.State import SearchState


def route(state: SearchState):
    if state["step"] == "retry_search":
        return "search"
    elif state["step"] == "tool_called":
        return "tools"
    return END


def create_search_assistant():
    workflow = StateGraph(SearchState)

    # 添加节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    workflow.add_node("sorcery", sorcery_answer_node)
    workflow.add_node("tools", tools_node)

    # 设置流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_conditional_edges(
        "answer",
        route,
        {
            "tools": "tools",
            "sorcery": "sorcery",
        }
    )
    workflow.add_edge("tools", "answer")

    workflow.add_edge("answer", "sorcery")
    workflow.add_conditional_edges(
        "sorcery",
        route,
        {
            "search": "search",
            END: END
        }
    )

    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app
