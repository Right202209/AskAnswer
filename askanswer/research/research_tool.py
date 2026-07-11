"""把 research 子图封装成 LLM 可调用的 Tool。

与 sql_tool / helix_tool 同款：被 react 子图的 ``ToolNode`` 调用时，``runtime`` 会自动
注入，子图内部就能拿到与父图一致的 ``ContextSchema``。
"""

from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ..schema import ContextSchema, normalize_context
from .agent import MAX_QUERIES, extract_research_answer, run_research_agent


@tool
def research_brief_loop(
    topic: str,
    runtime: ToolRuntime[ContextSchema],
    max_queries: int = MAX_QUERIES,
) -> str:
    """多源检索 + 交叉核验 + 列引用的研究简报循环。

    适用：用户想了解“最近某政策/产品/事件的变化”，需要综合多个来源、给出可追溯
    引用，而不是仅凭单次搜索作答。

    参数:
        topic: 用户的研究主题（自然语言）。
        max_queries: 最多规划的检索关键词数（会被收敛到 1..5）。

    返回:
        Markdown 文本，含 Research Brief / References / Queries 三块。
    """
    context = normalize_context(getattr(runtime, "context", None))
    result = run_research_agent(topic, max_queries=max_queries, context=context)
    return extract_research_answer(result)
