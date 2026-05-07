"""Runtime registry for intent handlers."""

from __future__ import annotations

from collections.abc import Iterable

from .base import IntentClassification, IntentHandler, extract_file_path
from .chat import CHAT_STARTERS, ChatHandler
from .file_read import FileReadHandler
from .helix import HelixHandler
from .math import MathHandler
from .search import SearchHandler
from .sql import SqlHandler


class IntentRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, IntentHandler] = {}

    def register(self, handler: IntentHandler) -> None:
        self._handlers[handler.name] = handler

    def get(self, name: str | None) -> IntentHandler:
        intent = str(name or "").lower().strip()
        return self._handlers.get(intent) or self._handlers["search"]

    def names(self) -> set[str]:
        return set(self._handlers)

    def handlers(self) -> list[IntentHandler]:
        return sorted(self._handlers.values(), key=lambda h: (h.priority, h.name))

    def tool_tags(self, name: str | None) -> frozenset[str]:
        return self.get(name).bundle_tags

    def classify_local(self, text: str, *, fallback: bool = False) -> IntentClassification | None:
        clean = str(text or "").strip()
        for handler in self.handlers():
            fields = handler.local_classify(clean)
            if fields is not None:
                return self.normalize(fields, clean)
        if not fallback:
            return None
        return self.normalize(self._fallback_classification(clean), clean)

    def normalize(
        self,
        fields: IntentClassification | dict,
        user_message: str,
    ) -> IntentClassification:
        if not isinstance(fields, IntentClassification):
            fields = IntentClassification.model_validate(fields)
        intent = fields.intent if fields.intent in self._handlers else "search"
        file_path = fields.file_path
        if intent == "file_read" and not file_path:
            file_path = extract_file_path(user_message)
        search_query = fields.search_query
        if intent == "search" and not search_query:
            search_query = user_message
        elif intent != "search":
            search_query = ""
        if intent != "file_read":
            file_path = ""
        return IntentClassification(
            intent=intent,
            file_path=file_path,
            search_query=search_query,
            understanding=fields.understanding or user_message,
        )

    def llm_intent_list(self) -> str:
        return "|".join(self.names())

    def _fallback_classification(self, text: str) -> IntentClassification:
        lowered = text.lower()
        file_path = extract_file_path(text)
        if file_path:
            return IntentClassification(
                intent="file_read",
                file_path=file_path,
                understanding=f"读取或分析本地文件：{file_path}",
            )
        intent = "chat"
        if len(text) > 80 and "?" not in text and "？" not in text:
            intent = "search"
        elif any(lowered.startswith(starter) for starter in CHAT_STARTERS):
            intent = "chat"
        elif not text:
            intent = "chat"
        return IntentClassification(
            intent=intent,
            search_query=text if intent == "search" else "",
            understanding=text,
        )


_registry = IntentRegistry()
for _handler in (
    FileReadHandler(),
    SqlHandler(),
    HelixHandler(),
    MathHandler(),
    SearchHandler(),
    ChatHandler(),
):
    _registry.register(_handler)


def get_intent_registry() -> IntentRegistry:
    return _registry


def register(handler: IntentHandler) -> None:
    _registry.register(handler)


def registered_handlers() -> Iterable[IntentHandler]:
    return _registry.handlers()
