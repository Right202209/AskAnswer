"""进程内事件总线（架构文档 6.4）。

业务代码只 ``publish(Event)``，不关心谁在听：审计落库、外部 telemetry、
本轮用量统计都是订阅者。订阅者抛异常不会影响业务流程。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

_LOG = logging.getLogger(__name__)

# 视为「一次工具执行」的事件：普通工具调用 + 审批类工具的执行结果
TOOL_EVENT_SUFFIXES = ("_approve", "_reject")


@dataclass(frozen=True)
class Event:
    kind: str                                   # llm_call / tool_call / shell_approve / model_swap ...
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    @property
    def is_tool_event(self) -> bool:
        return self.kind == "tool_call" or self.kind.endswith(TOOL_EVENT_SUFFIXES)


Subscriber = Callable[[Event], None]


class EventBus:
    """同步发布订阅。"""

    def __init__(self) -> None:
        self._subs: list[Subscriber] = []
        self._lock = Lock()

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        """注册订阅者，返回取消订阅函数。"""
        with self._lock:
            self._subs.append(fn)

        def _unsubscribe() -> None:
            with self._lock:
                if fn in self._subs:
                    self._subs.remove(fn)

        return _unsubscribe

    def publish(self, event: Event) -> None:
        with self._lock:
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - 订阅者故障不影响业务
                _LOG.exception("event subscriber failed: %s", event.kind)


def telemetry_sink(event: Event) -> None:
    """把工具执行事件转发给外部 telemetry（LangSmith / OTEL）。未启用时零开销。"""
    if not event.is_tool_event:
        return
    from askanswer.observability import telemetry

    telemetry.emit_event(
        kind="tool_call",
        tool_name=event.data.get("tool_name"),
        duration_ms=event.data.get("duration_ms"),
        error=event.data.get("error"),
    )


_default_bus: EventBus | None = None
_default_lock = Lock()


def get_bus() -> EventBus:
    """进程默认总线：首次访问时挂上审计与 telemetry 两个订阅者。"""
    global _default_bus
    with _default_lock:
        if _default_bus is None:
            from askanswer.observability.audit import audit_sink

            bus = EventBus()
            bus.subscribe(audit_sink)
            bus.subscribe(telemetry_sink)
            _default_bus = bus
        return _default_bus


def publish(kind: str, **data: Any) -> None:
    get_bus().publish(Event(kind=kind, data=data))
