"""research_brief 子图各节点：plan_queries → search → synthesize → source_check。

设计与 helix / sql 子图一致：节点只返回部分状态字典；与 LLM 的强类型交互走
``model.with_structured_output``。检索刻意**通过 ToolRegistry 调用 ``tavily_search``**
而非直连 tavily client —— 便于将来替换搜索后端，也复用统一的工具治理。
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from ..load import model

# 单次 research 调用允许规划/执行的最大检索次数，避免把 token 和时延拉爆。
MAX_QUERIES = 5
# 每条 tavily 结果里最多抽取的来源链接数
_MAX_SOURCES_PER_QUERY = 5
# 从 tavily 结果文本里抽取来源链接的正则（与 tavily_search 的 "链接: url" 输出对齐）
_SOURCE_URL_RE = re.compile(r"链接:\s*(\S+)")


class ResearchState(MessagesState):
    """research 子图状态：在标准消息状态上扩展检索计划与结果。"""
    topic: str
    max_queries: int
    queries: list[str]
    findings: list[dict]      # [{"query": str, "result": str, "sources": [url, ...]}]
    brief: str
    references: list[str]


# ── Pydantic Schemas ────────────────────────────────────────────────

class PlanOutput(BaseModel):
    queries: list[str] = Field(
        description="覆盖不同角度的检索关键词，彼此尽量不重复；2-5 条最佳",
        min_length=1,
        max_length=8,
    )


class SourceCheckOutput(BaseModel):
    brief: str = Field(description="交叉核验后的最终简报正文，保留 [n] 行内引用标注")
    references: list[str] = Field(
        default_factory=list,
        description="按 [n] 顺序排列的来源 URL 列表；无来源时给空列表",
    )


def _structured(schema):
    return model.with_structured_output(schema)


# ── Nodes ───────────────────────────────────────────────────────────

def plan_queries_node(state: ResearchState) -> dict:
    """把研究主题拆成多条互补的检索关键词。"""
    topic = state.get("topic") or ""
    limit = _clamp_queries(state.get("max_queries"))
    system = SystemMessage(content=(
        "你是研究规划员。针对用户主题，给出覆盖不同角度、彼此不重复的检索关键词，"
        f"最多 {limit} 条。优先覆盖：最新变化、权威来源、正反观点。"
    ))
    user = HumanMessage(content=f"研究主题：{topic}")
    output: PlanOutput = _structured(PlanOutput).invoke([system, user])
    queries = [q.strip() for q in output.queries if q.strip()][:limit]
    return {
        "queries": queries,
        "messages": [AIMessage(content=f"research 规划了 {len(queries)} 条检索关键词。")],
    }


def search_node(state: ResearchState) -> dict:
    """逐条调用 registry 里的 tavily_search，收集结果与来源链接。"""
    queries = state.get("queries") or []
    findings: list[dict] = []
    for query in queries:
        result = _run_search(query)
        findings.append(
            {"query": query, "result": result, "sources": _extract_sources(result)}
        )
    return {
        "findings": findings,
        "messages": [AIMessage(content=f"research 完成 {len(findings)} 次检索。")],
    }


def synthesize_node(state: ResearchState) -> dict:
    """把多源检索结果综合成带行内引用的研究简报草稿。"""
    topic = state.get("topic") or ""
    findings = state.get("findings") or []
    system = SystemMessage(content=(
        "你是研究综合员。基于多源检索结果撰写结构化简报：先给要点结论，再展开。"
        "对每条关键论断用 [n] 标注来源（n 对应下面来源清单的序号）；"
        "不要编造来源，检索里没有的结论要显式标注为推测。"
    ))
    user = HumanMessage(content=f"研究主题：{topic}\n\n{_format_findings(findings)}")
    response = model.invoke([system, user])
    return {
        "brief": str(getattr(response, "content", "") or ""),
        "messages": [AIMessage(content="research 已产出简报草稿。")],
    }


def source_check_node(state: ResearchState) -> dict:
    """交叉核验草稿与来源，剔除无支撑论断并整理最终引用清单。"""
    brief = state.get("brief") or ""
    findings = state.get("findings") or []
    system = SystemMessage(content=(
        "你是事实核查员。逐条核对简报里的 [n] 论断是否被对应来源支撑：无支撑的下调为"
        "'（待证实）'。最后按 [n] 顺序输出去重后的来源 URL 列表。"
    ))
    user = HumanMessage(content=(
        f"简报草稿：\n{brief}\n\n可用来源：\n{_format_sources(findings)}"
    ))
    output: SourceCheckOutput = _structured(SourceCheckOutput).invoke([system, user])
    return {
        "brief": output.brief,
        "references": [r.strip() for r in output.references if r.strip()],
        "messages": [AIMessage(content="research 已完成交叉核验。")],
    }


# ── helpers ─────────────────────────────────────────────────────────

def _clamp_queries(value) -> int:
    """把 max_queries 收敛到 [1, MAX_QUERIES]，非法输入回退默认上限。"""
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return MAX_QUERIES
    return max(1, min(limit, MAX_QUERIES))


def _run_search(query: str) -> str:
    """通过 ToolRegistry 调用 tavily_search（lazy import 避免 import 期循环）。"""
    from ..registry import get_registry

    descriptor = get_registry().get("tavily_search")
    if descriptor is None:
        return f"（tavily_search 工具不可用，跳过检索：{query}）"
    try:
        return str(descriptor.tool.invoke({"query": query}))
    except Exception as exc:
        return f"（检索失败：{exc}）"


def _extract_sources(result_text: str) -> list[str]:
    """从 tavily 结果文本里抽取来源链接，去重并限量。"""
    seen: list[str] = []
    for url in _SOURCE_URL_RE.findall(result_text or ""):
        if url not in seen:
            seen.append(url)
        if len(seen) >= _MAX_SOURCES_PER_QUERY:
            break
    return seen


def _format_findings(findings: list[dict]) -> str:
    if not findings:
        return "（无检索结果）"
    blocks = []
    for item in findings:
        blocks.append(f"### 关键词：{item.get('query', '')}\n{item.get('result', '')}")
    return "\n\n".join(blocks)


def _format_sources(findings: list[dict]) -> str:
    urls: list[str] = []
    for item in findings:
        for url in item.get("sources") or []:
            if url not in urls:
                urls.append(url)
    if not urls:
        return "（无可用来源链接）"
    return "\n".join(f"[{i}] {url}" for i, url in enumerate(urls, 1))
