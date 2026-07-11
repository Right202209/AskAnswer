"""Decision intent handler：触发决策备忘子图工具。

触发口径：显式的“帮我做决策 / 选型 / 对比方案给建议”类请求。粒度保守 —— 只在明显是
“要在几个方向里权衡取舍”时切到 decision，其它模糊需求仍交给 Helix 澄清。
"""

from __future__ import annotations

from ..state import SearchState
from .base import EvaluationResult, IntentClassification, pass_result


DECISION_KEYWORDS_LOWER = (
    "decision memo",
    "trade-off",
    "tradeoff",
    "trade off",
    "pros and cons",
    "which should i choose",
    "help me decide",
    "help me choose",
    "recommendation between",
)
DECISION_KEYWORDS_CN = (
    "决策备忘",
    "帮我决策",
    "帮我做决定",
    "帮我选",
    "怎么选",
    "如何选择",
    "选型",
    "权衡",
    "取舍",
    "利弊",
    "优缺点对比",
    "给个建议选",
)


class DecisionHandler:
    name = "decision"
    # 介于 sql(20) 与 helix(22) 之间：显式的“决策/选型/权衡”关键词应优先于 Helix
    # 的模糊需求启发式（"做个X"），避免 "帮我做个选型决策" 被 Helix 抢走。
    priority = 21
    bundle_tags = frozenset({"decision", "decision_tool", "helix", "search"})
    # 子图内部 interview + decide 已足够；父 sorcery 不再重试（Helix 同款理由）。
    max_retries = 0

    def local_classify(self, text: str) -> IntentClassification | None:
        clean = str(text or "").strip()
        if not clean:
            return None
        lowered = clean.lower()
        if any(keyword in lowered for keyword in DECISION_KEYWORDS_LOWER):
            return IntentClassification(intent=self.name, understanding=clean)
        if any(keyword in clean for keyword in DECISION_KEYWORDS_CN):
            return IntentClassification(intent=self.name, understanding=clean)
        return None

    def prompt_hint(self, state: SearchState) -> str:
        return (
            "（这是决策请求：不要直接给结论，先调用 decision_memo_loop 工具，"
            "传入用户原始决策作为 topic，等子图返回带取舍分析的 Markdown 备忘后再回答。）"
        )

    def evaluate(self, state: SearchState) -> EvaluationResult:
        # 子图自带 interview + decide；父图不重复评估。
        return pass_result("decision subgraph self-contained")

    def cli_label(self, update: dict) -> str:
        return "decision"
