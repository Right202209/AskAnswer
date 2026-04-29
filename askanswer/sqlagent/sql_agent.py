from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from ..schema import ContextSchema, normalize_context
from .sql_node import (
    call_get_schema,
    check_query,
    generate_query,
    get_schema_node,
    list_tables,
    run_query_node,
)


MAX_SQL_QUERY_CALLS = 2
SQL_RECURSION_LIMIT = 12


class SqlAgentState(MessagesState):
    query_count: int


def should_continue(state: SqlAgentState) -> Literal[END, "check_query", "limit_exceeded"]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls:
        return END
    if state.get("query_count", 0) >= MAX_SQL_QUERY_CALLS:
        return "limit_exceeded"
    return "check_query"


def limit_exceeded(state: SqlAgentState) -> dict:
    latest_result = ""
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and getattr(message, "name", "") == "sql_db_query":
            latest_result = str(message.content)
            break

    detail = f"\n\n最近一次查询结果：\n{latest_result}" if latest_result else ""
    message = (
        f"SQL agent 已达到查询上限（最多 {MAX_SQL_QUERY_CALLS} 次查询工具调用），"
        "已停止继续重试以避免超时或 502。请缩小问题范围，或检查 SQL/数据库返回结果后重试。"
        f"{detail}"
    )
    return {"messages": [AIMessage(content=message)]}


def build_sql_agent():
    builder = StateGraph(SqlAgentState, context_schema=ContextSchema)
    builder.add_node("list_tables", list_tables)
    builder.add_node("call_get_schema", call_get_schema)
    builder.add_node("get_schema", get_schema_node)
    builder.add_node("generate_query", generate_query)
    builder.add_node("check_query", check_query)
    builder.add_node("run_query", run_query_node)
    builder.add_node("limit_exceeded", limit_exceeded)

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges(
        "generate_query",
        should_continue,
        {"check_query": "check_query", "limit_exceeded": "limit_exceeded", END: END},
    )
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")
    builder.add_edge("limit_exceeded", END)

    return builder.compile()


sql_agent = build_sql_agent()


def run_sql_agent(messages: list, context: ContextSchema | dict | None = None) -> list:
    result = sql_agent.invoke(
        {"messages": messages, "query_count": 0},
        config={"recursion_limit": SQL_RECURSION_LIMIT},
        context=normalize_context(context),
    )
    return result["messages"]


def extract_sql_answer(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "content", ""):
            return str(message.content)
    return "SQL agent 未生成答案。"
