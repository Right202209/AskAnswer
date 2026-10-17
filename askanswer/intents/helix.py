"""Helix intent handler：触发规格优先演化循环工具。

触发口径：
1. 关键字（``helix`` / ``苏格拉底`` / ``需求澄清`` / ``spec-first`` …）—— 老逻辑；
2. 模糊需求启发式 —— 含有 “做一个 / 搞个 / build a …” 这类模糊动词，但没有具体
   实体（路径 / SQL / 数学），这种典型的 “想做点什么但还没想清楚” 也走 Helix。
"""

from __future__ import annotations

import re

from askanswer.graph.state import SearchState
from askanswer.intents.base import (
    EvaluationResult,
    IntentClassification,
    ModelProvider,
    pass_result,
)

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

# 模糊动词：用户说 “想做一个 X / 搞个 Y” 但 X / Y 描述不具体，多半需要先澄清。
AMBIGUITY_VERBS_CN = (
    "做一个",
    "做个",
    "搞一个",
    "搞个",
    "弄一个",
    "弄个",
    "写一个",
    "写个",
    "开发一个",
    "开发个",
    "想做",
    "想搞",
    "想写",
    "帮我做",
    "帮我搞",
    "帮我写",
    "帮我开发",
)
AMBIGUITY_VERBS_EN = (
    "build a ",
    "build an ",
    "make a ",
    "make an ",
    "design a ",
    "design an ",
    "create a ",
    "create an ",
    "develop a ",
    "develop an ",
    "i want to build",
    "help me build",
)

# 当文本含 “模糊动词” 但同时命中下面这些“具体性标记”时，就不再认定为模糊需求 ——
# 让 sql / file_read / math / search 这些更精准的 handler 兜走。
SPECIFICITY_MARKERS = re.compile(
    r"""(?ix)
    \b(?:select|insert|update|delete|from|where|join)\b   # SQL 关键字
    | https?://                                            # URL
    | [./~][\w\-./]*\.(?:py|md|json|yaml|yml|csv|tsv|txt)  # 文件路径
    | [\d.]+\s*[\+\-\*/%]\s*[\d.]+                         # 算式
    """,
)


def _looks_ambiguous(text: str) -> bool:
    """启发式判断：是否一段“想做点什么但描述不具体”的需求。"""
    clean = text.strip()
    if not clean or len(clean) > 200:
        # 太长通常意味着已经包含了足够细节，不需要 Helix 来澄清。
        return False
    lowered = clean.lower()
    has_vague = any(v in clean for v in AMBIGUITY_VERBS_CN) or any(
        v in lowered for v in AMBIGUITY_VERBS_EN
    )
    if not has_vague:
        return False
    if SPECIFICITY_MARKERS.search(clean):
        # 含具体的 SQL / URL / 路径 / 算式：让更精准的 handler 处理。
        return False
    return True


class HelixHandler:
    name = "helix"
    # 供 LLM 分类提示使用的一句话描述；分类提示由注册表从各 handler 生成（P5）
    description = "需求模糊、不具体（“想做一个 X”但缺乏关键约束、受众、验收标准），需要先苏格拉底式澄清再动手；包含 spec-first / 需求澄清 / 苏格拉底等关键字时也属于此类"
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
        # 关键字没命中时再看是不是“模糊需求” —— 这是新增的“需求不明确即切换”逻辑。
        if _looks_ambiguous(clean):
            return IntentClassification(intent=self.name, understanding=clean)
        return None

    def prompt_hint(self, state: SearchState) -> str:
        return (
            "（这是规格优先开发请求；不要直接给方案，先调用 helix_spec_loop 工具，"
            "传入用户原始需求作为 topic，等子图返回 Markdown 规格摘要后再回答。）"
        )

    def evaluate(self, state: SearchState, models: ModelProvider | None = None) -> EvaluationResult:
        # 子图内部已经做过 evaluate 阶段；父图不重复评估。
        return pass_result("helix subgraph self-evaluates")

    def cli_label(self, update: dict) -> str:
        return "helix"
