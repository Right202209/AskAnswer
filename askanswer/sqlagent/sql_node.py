from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime

from ..load import model
from ..schema import ContextSchema, normalize_context
from .sql_interact import get_sql_dialect, get_sql_tool


MAX_TABLE_LIST_CHARS = 4000
MAX_SCHEMA_CHARS = 12000
MAX_QUERY_RESULT_CHARS = 8000


def _runtime_context(runtime: Runtime[ContextSchema]) -> ContextSchema:
    return normalize_context(getattr(runtime, "context", None))


def _tool(name: str, runtime: Runtime[ContextSchema]):
    return get_sql_tool(name, _runtime_context(runtime))


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[内容过长，已截断 {omitted} 个字符]"


def _trim_observation(tool_name: str, content: object) -> str:
    text = str(content)
    limits = {
        "sql_db_list_tables": MAX_TABLE_LIST_CHARS,
        "sql_db_schema": MAX_SCHEMA_CHARS,
        "sql_db_query": MAX_QUERY_RESULT_CHARS,
    }
    return _trim_text(text, limits.get(tool_name, MAX_QUERY_RESULT_CHARS))


def _user_question(state: MessagesState) -> str:
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            return str(message.content)
    return str(getattr(state["messages"][0], "content", "")) if state["messages"] else ""


def _latest_tool_content(state: MessagesState, tool_name: str) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and getattr(message, "name", "") == tool_name:
            return str(message.content)
    return ""


def _latest_sql_query(state: MessagesState) -> str:
    for message in reversed(state["messages"]):
        for tool_call in reversed(getattr(message, "tool_calls", None) or []):
            if tool_call.get("name") == "sql_db_query":
                args = tool_call.get("args") or {}
                return str(args.get("query") or "")
    return ""


def _schema_selection_messages(state: MessagesState) -> list:
    tables = _latest_tool_content(state, "sql_db_list_tables") or "未获取到表列表。"
    return [
        SystemMessage(
            content=(
                "根据用户问题和可用表，选择回答问题所需的最少表，"
                "并调用 sql_db_schema 获取这些表结构。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：{_user_question(state)}\n\n"
                f"可用表：\n{_trim_text(tables, MAX_TABLE_LIST_CHARS)}"
            )
        ),
    ]


def _query_generation_messages(state: MessagesState, system_message: SystemMessage) -> list:
    schema = _latest_tool_content(state, "sql_db_schema") or "未获取到表结构。"
    previous_query = _latest_sql_query(state)
    previous_result = _latest_tool_content(state, "sql_db_query")

    parts = [
        f"用户问题：{_user_question(state)}",
        f"数据库表结构：\n{_trim_text(schema, MAX_SCHEMA_CHARS)}",
    ]
    if previous_query:
        parts.append(f"上一条 SQL：\n{previous_query}")
    if previous_result:
        parts.append(f"上一条 SQL 的结果或错误：\n{_trim_text(previous_result, MAX_QUERY_RESULT_CHARS)}")

    return [system_message, HumanMessage(content="\n\n".join(parts))]


def _run_last_tool_calls(state: MessagesState, tool) -> dict:
    responses = []
    for tool_call in getattr(state["messages"][-1], "tool_calls", None) or []:
        requested_name = tool_call["name"]
        if requested_name != tool.name:
            result = f"SQL 工具名不匹配：期望 {tool.name}，收到 {requested_name}"
            response_name = requested_name
        else:
            result = tool.invoke(tool_call.get("args") or {})
            response_name = tool.name
        responses.append(
            ToolMessage(
                content=_trim_observation(response_name, result),
                tool_call_id=tool_call["id"],
                name=response_name,
            )
        )
    return {"messages": responses}


def get_schema_node(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    return _run_last_tool_calls(state, _tool("sql_db_schema", runtime))


def run_query_node(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    result = _run_last_tool_calls(state, _tool("sql_db_query", runtime))
    if result["messages"]:
        result["query_count"] = state.get("query_count", 0) + len(result["messages"])
    return result


def list_tables(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    list_tables_tool = _tool("sql_db_list_tables", runtime)
    tool_call_id = f"list_tables_{uuid4().hex}"
    tool_call = {
        "name": list_tables_tool.name,
        "args": {},
        "id": tool_call_id,
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    tool_result = list_tables_tool.invoke({})
    response = ToolMessage(
        content=_trim_observation(list_tables_tool.name, tool_result),
        tool_call_id=tool_call["id"],
        name=list_tables_tool.name,
    )
    return {"messages": [tool_call_message, response]}


def call_get_schema(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    get_schema_tool = _tool("sql_db_schema", runtime)
    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(_schema_selection_messages(state))
    return {"messages": [response]}


generate_query_system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
"""


def generate_query(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    system_message = SystemMessage(
        content=generate_query_system_prompt.format(
            dialect=get_sql_dialect(_runtime_context(runtime)),
            top_k=5,
        ),
    )
    run_query_tool = _tool("sql_db_query", runtime)
    llm_with_tools = model.bind_tools([run_query_tool])
    response = llm_with_tools.invoke(_query_generation_messages(state, system_message))
    return {"messages": [response]}


check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
"""


def check_query(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    system_message = {
        "role": "system",
        "content": check_query_system_prompt.format(
            dialect=get_sql_dialect(_runtime_context(runtime)),
        ),
    }
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    run_query_tool = _tool("sql_db_query", runtime)
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id
    return {"messages": [response]}
