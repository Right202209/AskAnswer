"""Research intent handler：触发多源研究简报子图工具。

触发口径：显式研究/调研类关键词，或“最近某事物有什么变化”这类需要联网多源核验的
问句。粒度偏保守 —— 只在明显是“要做一份带引用的研究简报”时才切到 research，否则让
更轻量的 search handler 兜走单次检索。
"""

from __future__ import annotations

from ..schema import ContextSchema
from ..state import SearchState
from .base import (
    ClarificationChoice,
    ClarificationRequest,
    EvaluationResult,
    IntentClassification,
    pass_result,
)


# user_query 短于此长度时，判定研究范围偏宽、值得先澄清聚焦角度（启发式，可调）。
RESEARCH_SCOPE_MIN_CHARS = 24


# 研究/调研类关键词（中英混合）。命中即优先走 research 子图而非单次 search。
RESEARCH_KEYWORDS_LOWER = (
    "research",
    "deep dive",
    "literature review",
    "cross-reference",
    "cross reference",
    "cite sources",
    "with citations",
    "market landscape",
    "competitive analysis",
)
RESEARCH_KEYWORDS_CN = (
    "研究简报",
    "调研",
    "研究报告",
    "综述",
    "多源",
    "交叉核验",
    "交叉验证",
    "列出来源",
    "给出引用",
    "带引用",
    "竞品分析",
    "市场调研",
)


class ResearchHandler:
    name = "research"
    # 介于 search(30) 之前：比泛化 search 更具体，落到这里前仍让 sql/file_read/helix
    # 等更精准的 handler 优先匹配。
    priority = 28
    bundle_tags = frozenset({"research", "search", "research_tool", "tavily"})
    # 子图内部已做 source_check 自校验；父图额外再给最多 2 次重试兜底。
    max_retries = 2

    def local_classify(self, text: str) -> IntentClassification | None:
        clean = str(text or "").strip()
        if not clean:
            return None
        lowered = clean.lower()
        if any(keyword in lowered for keyword in RESEARCH_KEYWORDS_LOWER):
            return IntentClassification(intent=self.name, understanding=clean)
        if any(keyword in clean for keyword in RESEARCH_KEYWORDS_CN):
            return IntentClassification(intent=self.name, understanding=clean)
        return None

    def prompt_hint(self, state: SearchState) -> str:
        return (
            "（这是研究简报请求：不要只搜一次，先调用 research_brief_loop 工具，"
            "传入用户原始主题作为 topic，等子图返回带引用的 Markdown 简报后再回答；"
            "回答末尾保留 References 引用块。）"
        )

    def clarify(
        self, state: SearchState, context: ContextSchema
    ) -> ClarificationRequest | None:
        """主题过宽时，先让用户挑一个聚焦角度，避免简报泛泛而谈。"""
        query = str(state.get("user_query") or "").strip()
        if not query or len(query) >= RESEARCH_SCOPE_MIN_CHARS:
            return None  # 空主题无从聚焦；足够具体则直接开跑
        # 默认项「全面概览」不改写 user_query = 保持现状，非 TTY 不回归；
        # 其余角度把聚焦提示拼进 user_query（answer 节点会带进 system prompt）。
        return ClarificationRequest(
            prompt=f"研究主题「{query}」范围偏宽，想聚焦到哪个角度？",
            choices=(
                ClarificationChoice(label="全面概览（不额外限定）"),
                ClarificationChoice(
                    label="最新进展（近一年动态）",
                    updates={"user_query": f"{query} —— 侧重最近一年的最新进展与变化"},
                ),
                ClarificationChoice(
                    label="原理与机制（深入分析）",
                    updates={"user_query": f"{query} —— 侧重技术原理、机制与深入分析"},
                ),
                ClarificationChoice(
                    label="对比与选型（优劣、替代方案）",
                    updates={"user_query": f"{query} —— 侧重横向对比、优劣与可选替代方案"},
                ),
            ),
            default_index=0,
        )

    def evaluate(self, state: SearchState) -> EvaluationResult:
        # 子图内部已交叉核验；父图仅在完全没产出简报时才可能重试。
        return pass_result("research subgraph self-checks sources")

    def cli_label(self, update: dict) -> str:
        return "research"
