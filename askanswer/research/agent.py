"""research_brief 子图：把 plan_queries → search → synthesize → source_check 编译成
独立子图。与 SQL / Helix 子图同款封装：``build_*`` 返回编译好的图，``run_*`` 是统一
入口，``format_*`` 把结果整理成带引用的 Markdown 简报。

子图是线性的（无自评循环），因此不需要额外的 recursion 上限。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from ..schema import ContextSchema, normalize_context
from .nodes import (
    MAX_QUERIES,
    ResearchState,
    plan_queries_node,
    search_node,
    source_check_node,
    synthesize_node,
)


def build_research_agent():
    """构建并编译 research 子图：plan_queries → search → synthesize → source_check。"""
    builder = StateGraph(ResearchState, context_schema=ContextSchema)

    builder.add_node("plan_queries", plan_queries_node)   # 规划多条检索关键词
    builder.add_node("search", search_node)               # 逐条检索，收集来源
    builder.add_node("synthesize", synthesize_node)       # 综合成带行内引用的草稿
    builder.add_node("source_check", source_check_node)   # 交叉核验并整理引用清单

    builder.add_edge(START, "plan_queries")
    builder.add_edge("plan_queries", "search")
    builder.add_edge("search", "synthesize")
    builder.add_edge("synthesize", "source_check")
    builder.add_edge("source_check", END)

    return builder.compile()


# 模块级单例：编译一次，多次复用。
research_agent = build_research_agent()


def run_research_agent(
    topic: str,
    max_queries: int = MAX_QUERIES,
    context: ContextSchema | dict | None = None,
) -> dict:
    """执行 research 子图并返回最终状态字典。

    ``topic`` 是用户的研究主题；``context`` 透传 ContextSchema（便于未来在节点里读
    tenant_id 等 per-invocation 配置）。
    """
    initial: ResearchState = {
        "messages": [HumanMessage(content=topic)],
        "topic": topic,
        "max_queries": max_queries,
        "queries": [],
        "findings": [],
        "brief": "",
        "references": [],
    }
    return research_agent.invoke(initial, context=normalize_context(context))


def format_research_brief(result: dict) -> str:
    """把子图结果整理成 Markdown 简报：正文 + 编号引用清单。"""
    brief = (result.get("brief") or "").strip() or "（未产出简报）"
    references = result.get("references") or []
    queries = result.get("queries") or []

    ref_block = (
        "\n".join(f"[{i}] {url}" for i, url in enumerate(references, 1))
        if references
        else "（无引用来源）"
    )
    query_line = " · ".join(queries) if queries else "（无）"
    return (
        f"## Research Brief\n{brief}\n\n"
        f"## References\n{ref_block}\n\n"
        f"## Queries\n{query_line}"
    )


def extract_research_answer(result: dict) -> str:
    """对外暴露的最终答案字符串：直接复用 Markdown 简报。"""
    return format_research_brief(result)


__all__ = [
    "MAX_QUERIES",
    "build_research_agent",
    "research_agent",
    "run_research_agent",
    "format_research_brief",
    "extract_research_answer",
]
