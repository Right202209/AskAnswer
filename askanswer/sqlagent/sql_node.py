# SQL 子图各节点的具体实现。
# 这里的核心思路：每个节点都尽量薄，复杂逻辑（截断、消息组装）抽成纯函数辅助。
from __future__ import annotations

import re
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.runtime import Runtime

from ..load import model
from ..schema import ContextSchema, normalize_context
from .sql_interact import get_sql_dialect, get_sql_tool


# 字符截断阈值：避免把超大表的 schema 或查询结果整段塞进 prompt
MAX_TABLE_LIST_CHARS = 4000
MAX_SCHEMA_CHARS = 12000
MAX_QUERY_RESULT_CHARS = 8000


# DML/DDL 关键字硬名单。SQL agent 只允许只读查询：即使模型偏航或 prompt 注入
# 让它生成 INSERT/UPDATE/DELETE/DROP/ALTER/...，也要在执行前直接拦截，而不是
# 只靠 prompt 里写一句 "DO NOT make DML" 那种软约束。
_FORBIDDEN_SQL_KEYWORDS = frozenset({
    "insert", "update", "delete", "drop", "truncate", "alter",
    "create", "replace", "grant", "revoke", "merge", "rename",
    "comment", "lock", "vacuum", "reindex", "cluster", "attach",
    "detach", "begin", "commit", "rollback", "savepoint",
})

# 注释清理：先去块注释 /* */ 再去行注释 -- 直到行尾
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.S)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_FIRST_TOKEN_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)")


def _check_sql_readonly(sql: str) -> str | None:
    """对单条/多条 SQL 做只读校验。命中破坏性关键字时返回该关键字，否则返回 None。"""
    if not sql or not sql.strip():
        return None
    cleaned = _BLOCK_COMMENT_RE.sub(" ", sql)
    cleaned = _LINE_COMMENT_RE.sub(" ", cleaned)
    # 用 ; 拆多语句；任意一条命中即拒绝整批
    for stmt in cleaned.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        match = _FIRST_TOKEN_RE.match(stmt)
        if not match:
            continue
        first = match.group(1).lower()
        # WITH ... AS (...) SELECT 也允许；只在 WITH 后跟 INSERT/UPDATE/DELETE 时拦
        if first == "with":
            tail = stmt[match.end():]
            inner = re.search(r"\)\s*([A-Za-z_]+)", tail)
            if inner and inner.group(1).lower() in _FORBIDDEN_SQL_KEYWORDS:
                return inner.group(1).lower()
            continue
        if first in _FORBIDDEN_SQL_KEYWORDS:
            return first
    return None


def _runtime_context(runtime: Runtime[ContextSchema]) -> ContextSchema:
    """从 runtime 上读 context，并做一次 normalize。"""
    return normalize_context(getattr(runtime, "context", None))


def _tool(name: str, runtime: Runtime[ContextSchema]):
    """按名字取对应 SQL 工具，自动带上当前 runtime 的 context（DSN 等）。"""
    return get_sql_tool(name, _runtime_context(runtime))


def _trim_text(text: str, limit: int) -> str:
    """超过 limit 时做硬截断，并在末尾标注被丢弃的字符数，方便排查。"""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n[内容过长，已截断 {omitted} 个字符]"


def _trim_observation(tool_name: str, content: object) -> str:
    """根据工具类型选择合适的截断阈值。"""
    text = str(content)
    limits = {
        "sql_db_list_tables": MAX_TABLE_LIST_CHARS,
        "sql_db_schema": MAX_SCHEMA_CHARS,
        "sql_db_query": MAX_QUERY_RESULT_CHARS,
    }
    return _trim_text(text, limits.get(tool_name, MAX_QUERY_RESULT_CHARS))


def _user_question(state: MessagesState) -> str:
    """从消息历史中找出第一条 HumanMessage 作为“用户原问”。"""
    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            return str(message.content)
    # 兜底：状态非空但里面没有 HumanMessage 时，至少给点东西
    return str(getattr(state["messages"][0], "content", "")) if state["messages"] else ""


def _latest_tool_content(state: MessagesState, tool_name: str) -> str:
    """倒序找最近一条来自指定工具的 ToolMessage 内容。"""
    for message in reversed(state["messages"]):
        if isinstance(message, ToolMessage) and getattr(message, "name", "") == tool_name:
            return str(message.content)
    return ""


def _latest_sql_query(state: MessagesState) -> str:
    """倒序找最近一次 sql_db_query 的入参 query 字符串。"""
    for message in reversed(state["messages"]):
        for tool_call in reversed(getattr(message, "tool_calls", None) or []):
            if tool_call.get("name") == "sql_db_query":
                args = tool_call.get("args") or {}
                return str(args.get("query") or "")
    return ""


def _schema_selection_messages(state: MessagesState) -> list:
    """构造让 LLM “根据问题挑表”这一步的对话消息。"""
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
    """构造让 LLM 生成 SQL 时的对话消息：包含 schema、上一条 SQL 与其结果。"""
    schema = _latest_tool_content(state, "sql_db_schema") or "未获取到表结构。"
    previous_query = _latest_sql_query(state)
    previous_result = _latest_tool_content(state, "sql_db_query")

    parts = [
        f"用户问题：{_user_question(state)}",
        f"数据库表结构：\n{_trim_text(schema, MAX_SCHEMA_CHARS)}",
    ]
    # 如果是“重写 SQL”的回合，把上一条 SQL 与其执行结果一并提供
    if previous_query:
        parts.append(f"上一条 SQL：\n{previous_query}")
    if previous_result:
        parts.append(f"上一条 SQL 的结果或错误：\n{_trim_text(previous_result, MAX_QUERY_RESULT_CHARS)}")

    return [system_message, HumanMessage(content="\n\n".join(parts))]


def _run_last_tool_calls(state: MessagesState, tool) -> dict:
    """执行最后一条 AIMessage 中的所有 tool_calls，并把结果包成 ToolMessage 返回。"""
    responses = []
    for tool_call in getattr(state["messages"][-1], "tool_calls", None) or []:
        requested_name = tool_call["name"]
        # 防御：LLM 偶尔会调错工具名，这里给出明确报错信息
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
    """执行 sql_db_schema 工具：把 LLM 选出的若干表的字段信息拉回来。"""
    return _run_last_tool_calls(state, _tool("sql_db_schema", runtime))


def run_query_node(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    """真正执行 SQL 的节点；同时累加 query_count 用于上限判断。

    在调用底层 sql_db_query 之前，做一次只读校验：命中 INSERT/UPDATE/DELETE/DROP
    等破坏性关键字直接合成一条拒绝 ToolMessage 返回，不下放到数据库执行。
    """
    sql_tool = _tool("sql_db_query", runtime)
    last_message = state["messages"][-1]
    tool_calls = list(getattr(last_message, "tool_calls", None) or [])
    responses: list[ToolMessage] = []
    for tool_call in tool_calls:
        requested_name = tool_call["name"]
        if requested_name != sql_tool.name:
            content = f"SQL 工具名不匹配：期望 {sql_tool.name}，收到 {requested_name}"
            responses.append(
                ToolMessage(
                    content=_trim_observation(requested_name, content),
                    tool_call_id=tool_call["id"],
                    name=requested_name,
                )
            )
            continue
        query = str((tool_call.get("args") or {}).get("query") or "")
        forbidden = _check_sql_readonly(query)
        if forbidden:
            content = (
                f"已拦截破坏性 SQL（{forbidden.upper()}）：本系统仅允许只读查询。\n"
                f"原 SQL：{query}"
            )
            responses.append(
                ToolMessage(
                    content=_trim_observation(sql_tool.name, content),
                    tool_call_id=tool_call["id"],
                    name=sql_tool.name,
                )
            )
            continue
        result = sql_tool.invoke(tool_call.get("args") or {})
        responses.append(
            ToolMessage(
                content=_trim_observation(sql_tool.name, result),
                tool_call_id=tool_call["id"],
                name=sql_tool.name,
            )
        )
    out: dict = {"messages": responses}
    if responses:
        # 一次 AIMessage 可能并发触发多条 sql_db_query，按消息条数累加
        out["query_count"] = state.get("query_count", 0) + len(responses)
    return out


def list_tables(state: MessagesState, runtime: Runtime[ContextSchema]) -> dict:
    """子图入口：直接调用 sql_db_list_tables，并把结果以一条 AIMessage + ToolMessage 写入历史。

    与其它节点不同，这一步不依赖 LLM —— 我们手工伪造 tool_call 然后调用工具，
    省一次 LLM 调用，让流水线快一点。
    """
    list_tables_tool = _tool("sql_db_list_tables", runtime)
    # 自己造一个 tool_call_id，保证它和 ToolMessage 一一对应
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
    """让 LLM 决定取哪些表的 schema：``tool_choice="any"`` 强制必须发起一次工具调用。"""
    get_schema_tool = _tool("sql_db_schema", runtime)
    llm_with_tools = model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(_schema_selection_messages(state))
    return {"messages": [response]}


# 生成 SQL 用的 system prompt：英文是为了与底层模型表现稳定一致，且与 LangChain 通用范式贴合
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
    """让 LLM 写 SQL：把方言、Top-K 注入 prompt，再绑定 sql_db_query 工具发起调用。"""
    system_message = SystemMessage(
        content=generate_query_system_prompt.format(
            dialect=get_sql_dialect(_runtime_context(runtime)),
            top_k=5,
        ),
    )
    run_query_tool = _tool("sql_db_query", runtime)
    # 这一步只让 LLM “提议”一条 SQL（即一次 tool_call），下一步 check_query 再做复核
    llm_with_tools = model.bind_tools([run_query_tool])
    response = llm_with_tools.invoke(_query_generation_messages(state, system_message))
    return {"messages": [response]}


# 校验 SQL 用的 system prompt：列出常见错误模式，让 LLM 自我审查再决定是否改写
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
    """让 LLM 复核上一步生成的 SQL，必要时改写后再调用 sql_db_query 工具。"""
    system_message = {
        "role": "system",
        "content": check_query_system_prompt.format(
            dialect=get_sql_dialect(_runtime_context(runtime)),
        ),
    }
    # 取出 generate_query 节点产生的待执行 SQL 作为用户消息喂给 LLM
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    run_query_tool = _tool("sql_db_query", runtime)
    # tool_choice="any" 强制 LLM 一定要发起一次 tool_call，避免它“只口头说没问题”
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    # 把 id 抹平成原 AIMessage 的 id，让消息历史中只保留校验后的版本
    response.id = state["messages"][-1].id
    return {"messages": [response]}
