"""Smoke-test intent proving new handlers can be registered without core edits."""

from __future__ import annotations

import re

from ..state import SearchState
from .base import EvaluationResult, IntentClassification, pass_result


MATH_RE = re.compile(r"(?<!\w)(?:\d+(?:\.\d+)?\s*[-+*/%()]+[\d\s.+\-*/%()]+)")
# 仅在「算」紧邻数字、等号或括号时才视为计算请求，避免命中 打算/估算/算了 等无关词。
COMPUTE_VERB_RE = re.compile(r"算[\s]*[\d=（(]")
EXPLICIT_COMPUTE = ("计算", "算式", "算出", "求值", "等于多少")


class MathHandler:
    name = "math"
    priority = 25
    bundle_tags = frozenset({"math"})
    max_retries = 0

    def local_classify(self, text: str) -> IntentClassification | None:
        clean = str(text or "").strip()
        lowered = clean.lower()
        if (
            any(keyword in clean for keyword in EXPLICIT_COMPUTE)
            or "calculate" in lowered
            or MATH_RE.search(clean)
            or COMPUTE_VERB_RE.search(clean)
        ):
            return IntentClassification(intent=self.name, understanding=clean)
        return None

    def prompt_hint(self, state: SearchState) -> str:
        return "（这是数学计算请求；需要精确计算时调用 calculate 工具。）"

    def evaluate(self, state: SearchState) -> EvaluationResult:
        return pass_result("math does not need post-answer evaluation")

    def cli_label(self, update: dict) -> str:
        return "math"
