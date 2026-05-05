"""把 SQL 代理子图封装成一个可被 LLM 调用的 Tool。

当此工具在 react 子图里被 ``ToolNode`` 调用时，``runtime`` 参数会自动注入，
内部 ``run_sql_agent`` 调用就能拿到与父图一致的 ``ContextSchema``
（db_dsn / dialect / tenant 等都自动透传）。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# ``ToolRuntime`` 来自 langchain.tools（langchain >= 0.3）。
# 它是 @tool 装饰器识别的“魔法参数名”，因此 LLM 看不到 runtime 这个参数 schema。
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
    # 兼容父图传入 dict 或 ContextSchema 两种形态
    context = normalize_context(getattr(runtime, "context", None))
    # 把用户问题包装成 HumanMessage 喂给 SQL 子图，最终拿回完整消息列表
    messages = run_sql_agent([HumanMessage(content=question)], context=context)
    # 从消息列表中提取最后一条 AIMessage 的内容作为答案
    return extract_sql_answer(messages)
