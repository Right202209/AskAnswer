# SQL 代理子图：把 list_tables → call_get_schema → get_schema → generate_query
# →（check_query → run_query → generate_query）* 这一套循环编译为独立子图。
# 通过硬上限 MAX_SQL_QUERY_CALLS / SQL_RECURSION_LIMIT 防止无限重试导致 502。
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


# 单次 sql_query 调用允许执行 SQL 的最大次数（避免反复重写 SQL 把成本拉爆）
MAX_SQL_QUERY_CALLS = 2
# LangGraph 的递归层数上限，作为最后一道保险
SQL_RECURSION_LIMIT = 12


class SqlAgentState(MessagesState):
    """SQL 子图的状态：在标准消息状态上额外加一个查询计数。"""
    query_count: int


def should_continue(state: SqlAgentState) -> Literal[END, "check_query", "limit_exceeded"]:
    """generate_query 之后的路由：决定继续校验/执行、放弃，还是直接结束。"""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    # LLM 没有继续要求执行 SQL 时直接收尾
    if not tool_calls:
        return END
    # 触达执行上限 → 走兜底节点，把现有的最后一次结果返回给上层
    if state.get("query_count", 0) >= MAX_SQL_QUERY_CALLS:
        return "limit_exceeded"
    return "check_query"


def limit_exceeded(state: SqlAgentState) -> dict:
    """达到 SQL 执行上限的兜底节点：返回最近一次结果而非简单报错。"""
    latest_result = ""
    # 倒序扫一下消息历史，找到最后一条 sql_db_query 的 ToolMessage
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
    """构建并编译 SQL 子图：典型的“先看表 → 取 schema → 出 SQL → 检查 → 执行”循环。"""
    builder = StateGraph(SqlAgentState, context_schema=ContextSchema)
    # 流水线节点：每一个都对应文件 sql_node.py 中的一个函数
    builder.add_node("list_tables", list_tables)             # 列出所有可用表
    builder.add_node("call_get_schema", call_get_schema)     # 让 LLM 决定取哪些表的 schema
    builder.add_node("get_schema", get_schema_node)          # 真正执行 sql_db_schema 工具
    builder.add_node("generate_query", generate_query)       # 让 LLM 写 SQL
    builder.add_node("check_query", check_query)             # 让 LLM 复核 SQL，必要时改写
    builder.add_node("run_query", run_query_node)            # 执行 SQL 并把结果回填
    builder.add_node("limit_exceeded", limit_exceeded)       # 达到上限的兜底

    # 起始线性段
    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    # generate_query 后根据是否还要执行 SQL 来路由
    builder.add_conditional_edges(
        "generate_query",
        should_continue,
        {"check_query": "check_query", "limit_exceeded": "limit_exceeded", END: END},
    )
    # 校验后必然执行；执行后回到 generate_query，让 LLM 看着结果决定收尾还是再试一轮
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")
    builder.add_edge("limit_exceeded", END)

    return builder.compile()


# 模块级单例：编译一次，多次复用
sql_agent = build_sql_agent()


def run_sql_agent(messages: list, context: ContextSchema | dict | None = None) -> list:
    """执行 SQL 子图并返回完整消息列表。"""
    result = sql_agent.invoke(
        {"messages": messages, "query_count": 0},
        # 设置递归上限，作为兜底保护
        config={"recursion_limit": SQL_RECURSION_LIMIT},
        # 把父图传来的 ContextSchema 透传给子图（包含 db_dsn / dialect 等）
        context=normalize_context(context),
    )
    return result["messages"]


def extract_sql_answer(messages: list) -> str:
    """从消息列表中倒序找出最后一条非空 AIMessage 的 content 作为最终答案。"""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "content", ""):
            return str(message.content)
    return "SQL agent 未生成答案。"
