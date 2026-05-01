"""Wrap the SQL agent subgraph as a tool the model can call.

When invoked from the react subgraph via ``ToolNode``, the ``runtime`` argument
is auto-injected so the inner ``run_sql_agent`` call receives the same
``ContextSchema`` (db_dsn / dialect / tenant) as the parent invocation.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# ``ToolRuntime`` lives in ``langchain.tools`` (langchain >= 0.3). It is a
# reserved parameter name recognized by the @tool decorator, so the LLM never
# sees it in the tool schema.
from langchain.tools import ToolRuntime

from ..schema import ContextSchema, normalize_context
from .sql_agent import extract_sql_answer, run_sql_agent


@tool
def sql_query(question: str, runtime: ToolRuntime[ContextSchema]) -> str:
    """用自然语言查询数据库。

    适用场景：用户希望统计、列出、聚合或分析数据库里的数据。需要 runtime
    context 中提供 ``db_dsn``。返回 SQL agent 整理后的中文答案。

    参数:
        question: 自然语言问题，例如 "上个月订单总数" 或 "用户表前 5 行"。
    """
    context = normalize_context(getattr(runtime, "context", None))
    messages = run_sql_agent([HumanMessage(content=question)], context=context)
    return extract_sql_answer(messages)
