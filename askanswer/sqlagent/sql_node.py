from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from ..load import model
from .sql_interact import get_sql_dialect, get_sql_tool


get_schema_tool = get_sql_tool("sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = get_sql_tool("sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")

list_tables_tool = get_sql_tool("sql_db_list_tables")


def list_tables(state: MessagesState) -> dict:
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


def call_get_schema(state: MessagesState) -> dict:
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
""".format(
    dialect=get_sql_dialect(),
    top_k=5,
)


def generate_query(state: MessagesState) -> dict:
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
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
""".format(dialect=get_sql_dialect())


def check_query(state: MessagesState) -> dict:
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id
    return {"messages": [response]}