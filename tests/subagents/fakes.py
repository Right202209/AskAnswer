"""子 agent 测试用的假模型：按 schema 产出结构化结果，按队列产出纯文本回复。"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import AIMessage


class FakeStructuredModel:
    """``with_structured_output(schema).invoke()`` → ``factories[schema](messages)``。

    factory 可以是实例（每次原样返回）、可调用对象（按收到的消息生成），或异常实例（抛出）。
    ``invoke()`` 直接调用时按 ``texts`` 队列返回 ``AIMessage``。
    """

    def __init__(self, factories: dict | None = None, texts: list[str] | None = None):
        self.factories = dict(factories or {})
        self.texts = list(texts or [])
        self.calls: list[tuple[str, object]] = []

    def with_structured_output(self, schema):
        return _Bound(self, schema)

    def invoke(self, messages, **kwargs):
        self.calls.append(("text", messages))
        return AIMessage(content=self.texts.pop(0) if self.texts else "")


class _Bound:
    def __init__(self, parent: FakeStructuredModel, schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, messages, **kwargs):
        self.parent.calls.append((self.schema.__name__, messages))
        factory = self.parent.factories[self.schema]
        if isinstance(factory, BaseException):
            raise factory
        if isinstance(factory, Callable) and not isinstance(factory, type):
            return factory(messages)
        return factory
