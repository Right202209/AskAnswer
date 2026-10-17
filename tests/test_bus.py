"""事件总线：业务只发布事件，审计 / telemetry 是订阅者。"""

from __future__ import annotations

from askanswer.observability import audit
from askanswer.observability.bus import Event, EventBus, get_bus


def test_subscriber_errors_do_not_break_publish():
    bus = EventBus()
    seen = []

    def boom(event):
        raise RuntimeError("x")

    bus.subscribe(boom)
    unsubscribe = bus.subscribe(seen.append)
    bus.publish(Event(kind="tool_call"))
    assert [e.kind for e in seen] == ["tool_call"]
    unsubscribe()
    bus.publish(Event(kind="tool_call"))
    assert len(seen) == 1


def test_log_event_goes_through_bus_and_is_buffered_per_run():
    seen = []
    unsubscribe = get_bus().subscribe(seen.append)
    tokens = audit.begin_run("thread-bus", tenant_id="t")
    try:
        audit.log_event(kind="tool_call", tool_name="pwd", duration_ms=3)
        audit.log_event(kind="llm_call", input_tokens=10, output_tokens=5)
        assert [e.kind for e in seen] == ["tool_call", "llm_call"]
        assert seen[0].is_tool_event and not seen[1].is_tool_event
        assert audit.run_usage_so_far() == (10, 5)  # 审计订阅者缓冲到本轮
    finally:
        audit.end_run(tokens)
        unsubscribe()


def test_tool_events_are_forwarded_to_telemetry(monkeypatch):
    from askanswer.observability import telemetry

    sent = []
    monkeypatch.setattr(telemetry, "emit_event", lambda **kw: sent.append(kw))
    audit.log_event(kind="shell_approve", tool_name="run_shell", duration_ms=7)
    audit.log_event(kind="llm_call", input_tokens=1)
    assert sent == [{"kind": "tool_call", "tool_name": "run_shell", "duration_ms": 7, "error": None}]
