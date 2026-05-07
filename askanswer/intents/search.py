"""Search intent handler."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from ..load import model
from ..state import SearchState
from .base import EvaluationResult, IntentClassification, latest_tool_message, pass_result


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
        search_results = state.get("search_results", "")
        if not search_results:
            return "（如需联网信息请调用 tavily_search 工具，否则基于已有知识回答。）"
        return f"以下是搜索结果，可作为参考：\n{search_results}"

    def evaluate(self, state: SearchState) -> EvaluationResult:
        final_answer = state.get("final_answer", "")
        tool_message = latest_tool_message(state, "tavily_search")
        search_results = state.get("search_results", "")
        if tool_message is not None:
            search_results = str(getattr(tool_message, "content", "") or search_results)

        prompt = f"""请评估当前答案是否足够好。
    用户问题：{state.get('user_query', '')}
    当前搜索词：{state.get('search_query', '')}
    当前搜索结果：{search_results}
    当前答案：{final_answer}

    如果答案已经足够好，请严格输出：
    评分：pass
    新搜索词：

    如果答案不够好，需要重新搜索，请严格输出：
    评分：re_search
    新搜索词：[更精准的新搜索词]
    """
        response = model.invoke([SystemMessage(content=prompt)])
        response_text = str(response.content or "")
        if "评分：re_search" not in response_text or "新搜索词：" not in response_text:
            return pass_result("evaluation passed")
        new_search_query = response_text.split("新搜索词：", 1)[1].strip()
        if not new_search_query:
            return pass_result("retry requested without query")
        current = str(state.get("search_query", "") or "").strip()
        if new_search_query == current:
            return pass_result("retry query duplicated current query")
        return EvaluationResult(
            decision="retry",
            retry_directive={
                "search_query": new_search_query,
                "instruction": f"前一次回答不够理想，请调用 tavily_search 工具用以下关键词重新搜索后再作答：{new_search_query}",
            },
            reason="search answer needs retry",
        )

    def cli_label(self, update: dict) -> str:
        query = update.get("search_query", "")
        return f"search: {query}" if query else "search"
