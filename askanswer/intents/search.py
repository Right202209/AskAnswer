"""Search intent handler."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from askanswer.graph.state import SLOT_SEARCH_QUERY, SearchState, get_slot
from askanswer.intents.base import (
    EvaluationResult,
    IntentClassification,
    ModelProvider,
    latest_tool_message,
    pass_result,
)
from askanswer.llm.routing import ROLE_EVALUATE, model_for


class SearchVerdict(BaseModel):
    """search 答案评估的结构化结果。"""

    decision: Literal["pass", "re_search"] = Field(
        description="答案足够好填 pass；需要换关键词重新搜索填 re_search"
    )
    new_search_query: str = Field(
        default="", description="decision=re_search 时给出更精准的新搜索词，否则留空"
    )


SEARCH_KEYWORDS = (
    "联网",
    "搜索",
    "搜一下",
    "查一下",
    "查找",
    "最新",
    "最近",
    "今天",
    "今日",
    "现在",
    "实时",
    "新闻",
    "价格",
    "股价",
    "汇率",
    "天气",
    "官网",
    "资料",
    "search",
    "latest",
    "recent",
    "today",
    "current",
    "realtime",
    "news",
    "price",
    "stock",
    "weather",
    "official",
)


class SearchHandler:
    name = "search"
    # 供 LLM 分类提示使用的一句话描述；分类提示由注册表从各 handler 生成（P5）
    description = "需要联网搜索获取实时、最新或不确定的信息"
    priority = 30
    bundle_tags = frozenset({"search"})
    max_retries = 2

    def local_classify(self, text: str) -> IntentClassification | None:
        clean = str(text or "").strip()
        lowered = clean.lower()
        if any(keyword in lowered for keyword in SEARCH_KEYWORDS):
            return IntentClassification(
                intent=self.name,
                search_query=clean,
                understanding=clean,
            )
        return None

    def prompt_hint(self, state: SearchState) -> str:
        query = get_slot(state, SLOT_SEARCH_QUERY)
        if query:
            return f"（如需联网信息请调用 tavily_search 工具，建议关键词：{query}；否则基于已有知识回答。）"
        return "（如需联网信息请调用 tavily_search 工具，否则基于已有知识回答。）"

    def evaluate(self, state: SearchState, models: ModelProvider | None = None) -> EvaluationResult:
        final_answer = state.get("final_answer", "")
        tool_message = latest_tool_message(state, "tavily_search")
        search_results = ""
        if tool_message is not None:
            search_results = str(getattr(tool_message, "content", "") or "")

        prompt = (
            "请评估当前答案是否足够好；如果不够好，给出一个更精准的新搜索词。\n"
            f"用户问题：{state.get('user_query', '')}\n"
            f"当前搜索词：{get_slot(state, SLOT_SEARCH_QUERY)}\n"
            f"当前搜索结果：{search_results}\n"
            f"当前答案：{final_answer}"
        )
        # 结构化输出代替在文本里找“评分：xxx”（架构文档 P7）：模型换个说法不会让判断失效
        try:
            verdict = (
                (models or model_for)(ROLE_EVALUATE)
                .with_structured_output(SearchVerdict)
                .invoke([SystemMessage(content=prompt)])
            )
        except Exception:
            return pass_result("evaluation failed; keep current answer")
        if not isinstance(verdict, SearchVerdict):
            verdict = SearchVerdict.model_validate(verdict)
        if verdict.decision != "re_search":
            return pass_result("evaluation passed")
        new_search_query = verdict.new_search_query.strip()
        if not new_search_query:
            return pass_result("retry requested without query")
        current = str(get_slot(state, SLOT_SEARCH_QUERY) or "").strip()
        if new_search_query == current:
            return pass_result("retry query duplicated current query")
        return EvaluationResult(
            decision="retry",
            retry_directive={
                "slots": {SLOT_SEARCH_QUERY: new_search_query},
                "instruction": f"前一次回答不够理想，请调用 tavily_search 工具用以下关键词重新搜索后再作答：{new_search_query}",
            },
            reason="search answer needs retry",
        )

    def cli_label(self, update: dict) -> str:
        query = get_slot(update, SLOT_SEARCH_QUERY)
        return f"search: {query}" if query else "search"
