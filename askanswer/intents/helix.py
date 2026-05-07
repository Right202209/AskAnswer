"""Helix intent handler：触发规格优先演化循环工具。"""

from __future__ import annotations

from ..state import SearchState
from .base import EvaluationResult, IntentClassification, pass_result


# 中英混合的关键字。粒度刻意偏粗：只要用户语义偏「想先把规格想清楚」就命中。
HELIX_KEYWORDS_LOWER = (
    "helix",
    "ouroboros",
    "spec-first",
    "spec first",
    "specification first",
    "specification-first",
    "socratic",
    "interview me",
    "acceptance criteria",
    "evolve loop",
    "evolutionary loop",
    "crystallize",
)
HELIX_KEYWORDS_CN = (
    "苏格拉底",
    "需求澄清",
    "澄清需求",
    "规格优先",
    "规格化",
    "演化循环",
    "迭代演化",
    "生成 seed",
    "生成seed",
    "晶化需求",
    "ontology",
    "acceptance",
)


class HelixHandler:
    name = "helix"
    # 介于 sql(20) 与 math(25) 之间：更具体的 SQL/文件意图先匹配，落到这里前还能
    # 兜住 chat / search 这些更宽泛的 fallback。
    priority = 22
    bundle_tags = frozenset({"helix"})
    # 子图自带最多 MAX_GENERATIONS 代演化，父图不再额外重试。
    max_retries = 0

    def local_classify(self, text: str) -> IntentClassification | None:
        clean = str(text or "").strip()
        if not clean:
            return None
        lowered = clean.lower()
        if any(keyword in lowered for keyword in HELIX_KEYWORDS_LOWER):
            return IntentClassification(intent=self.name, understanding=clean)
        if any(keyword in clean for keyword in HELIX_KEYWORDS_CN):
            return IntentClassification(intent=self.name, understanding=clean)
        return None

    def prompt_hint(self, state: SearchState) -> str:
        return (
            "（这是规格优先开发请求；不要直接给方案，先调用 helix_spec_loop 工具，"
            "传入用户原始需求作为 topic，等子图返回 Markdown 规格摘要后再回答。）"
        )

    def evaluate(self, state: SearchState) -> EvaluationResult:
        # 子图内部已经做过 evaluate 阶段；父图不重复评估。
        return pass_result("helix subgraph self-evaluates")

    def cli_label(self, update: dict) -> str:
        return "helix"
