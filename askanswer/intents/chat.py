"""Chat intent handler."""

from __future__ import annotations

from ..state import SearchState
from .base import EvaluationResult, IntentClassification, pass_result


CHAT_STARTERS = (
    "你好",
    "您好",
    "hello",
    "hi",
    "hey",
    "解释",
    "说明",
    "总结",
    "翻译",
    "改写",
    "写一段",
    "帮我写",
    "如何",
    "怎么",
    "为什么",
)


class ChatHandler:
    name = "chat"
    priority = 40
    bundle_tags = frozenset({"chat"})
    max_retries = 0

    def local_classify(self, text: str) -> IntentClassification | None:
        clean = str(text or "").strip()
        lowered = clean.lower()
        if any(lowered.startswith(starter) for starter in CHAT_STARTERS):
            return IntentClassification(intent=self.name, understanding=clean)
        return None

    def prompt_hint(self, state: SearchState) -> str:
        return "（这是闲聊或常识类问题，不需要搜索结果；可直接回答或调用合适的工具。）"

    def evaluate(self, state: SearchState) -> EvaluationResult:
        return pass_result("chat does not need post-answer evaluation")

    def cli_label(self, update: dict) -> str:
        return "chat"
