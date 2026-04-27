from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from .sql_node import (
    call_get_schema,
    check_query,
    generate_query,
    get_schema_node,
    list_tables,
    run_query_node,
)


class SqlAgentState(MessagesState):
    pass


def should_continue(state: SqlAgentState) -> Literal[END, "check_query"]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls:
        return END
    return "check_query"


def build_sql_agent():
    builder = StateGraph(SqlAgentState)
    builder.add_node("list_tables", list_tables)
    builder.add_node("call_get_schema", call_get_schema)
    builder.add_node("get_schema", get_schema_node)
    builder.add_node("generate_query", generate_query)
    builder.add_node("check_query", check_query)
    builder.add_node("run_query", run_query_node)

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")

    return builder.compile()


sql_agent = build_sql_agent()


def run_sql_agent(messages: list) -> list:
    result = sql_agent.invoke({"messages": messages})
    return result["messages"]


def extract_sql_answer(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "content", ""):
            return str(message.content)
    return "SQL agent 未生成答案。"