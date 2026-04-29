from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime

from ..load import model
from ..schema import ContextSchema, normalize_context
from .sql_interact import get_sql_dialect, get_sql_tool


def _runtime_context(runtime: Runtime[ContextSchema]) -> ContextSchema:
    return normalize_context(getattr(runtime, "context", None))


def _tool(name: str, runtime: Runtime[ContextSchema]):
    return get_sql_tool(name, _runtime_context(runtime))


def _run_last_tool_calls(state: MessagesState, tool) -> dict:
    responses = []
    for tool_call in getattr(state["messages"][-1], "tool_calls", None) or []:
        if tool_call["name"] != tool.name:
            result = f"SQL 工具名不匹配：期望 {tool.name}，收到 {tool_call['name']}"
        else:
            result = tool.invoke(tool_call.get("args") or {})
        responses.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool.name,
            )
        )
    return {"messages": responses}


def get_schema_node(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    return _run_last_tool_calls(state, _tool("sql_db_schema", runtime))


def run_query_node(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    return _run_last_tool_calls(state, _tool("sql_db_query", runtime))


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
        content=str(tool_result),
        tool_call_id=tool_call["id"],
        name=list_tables_tool.name,
    )
    return {"messages": [tool_call_message, response]}


def call_get_schema(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    get_schema_tool = _tool("sql_db_schema", runtime)
    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
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
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt.format(
            dialect=get_sql_dialect(_runtime_context(runtime)),
            top_k=5,
        ),
    }
    run_query_tool = _tool("sql_db_query", runtime)
    llm_with_tools = model.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])
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
