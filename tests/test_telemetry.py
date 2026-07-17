"""telemetry：关闭态零副作用、span 栈行为、事件派发到 exporter。"""

from __future__ import annotations

import pytest

import askanswer.telemetry as telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry():
    """每个用例独立初始化，避免模块级状态串扰。"""
    telemetry._initialized = False
    telemetry._exporters = []
    yield
    telemetry._initialized = False
    telemetry._exporters = []


class _FakeExporter:
    def __init__(self):
        self.events = []
        self.spans = []

    def emit(self, event):
        self.events.append(event)

    def start_span(self, span_id, name, parent_id, attrs):
        self.spans.append(("start", span_id, name, parent_id))

    def end_span(self, span_id, duration_ms, error):
        self.spans.append(("end", span_id, duration_ms, error))


def test_disabled_by_default_without_env():
    telemetry.init_telemetry()
    assert telemetry.is_enabled() is False
    telemetry.emit_event(kind="tool_call", tool_name="x")  # 必须是 no-op
    with telemetry.span("noop"):
        pass
    assert telemetry.llm_callback("gpt") is None
    assert telemetry.open_span("manual") is None
    telemetry.close_span(None)  # None handle 是 no-op


def test_init_is_idempotent(monkeypatch):
    calls = []
    monkeypatch.setattr(telemetry, "_load_exporters", lambda: calls.append(1) or [])
    telemetry.init_telemetry()
    telemetry.init_telemetry()
    assert len(calls) == 1


def test_emit_event_reaches_exporter():
    fake = _FakeExporter()
    telemetry._exporters = [fake]
    telemetry.emit_event(kind="tool_call", tool_name="t", duration_ms=5)
    assert len(fake.events) == 1
    assert fake.events[0]["tool_name"] == "t"
    assert fake.events[0]["span_id"] is None  # 无活跃 span


def test_span_nesting_sets_parent():
    fake = _FakeExporter()
    telemetry._exporters = [fake]
    with telemetry.span("outer"):
        with telemetry.span("inner"):
            telemetry.emit_event(kind="tool_call", tool_name="t")
    starts = [s for s in fake.spans if s[0] == "start"]
    ends = [s for s in fake.spans if s[0] == "end"]
    assert len(starts) == 2 and len(ends) == 2
    outer_id = starts[0][1]
    assert starts[0][3] is None  # outer 无 parent
    assert starts[1][3] == outer_id  # inner 的 parent 是 outer
    assert fake.events[0]["span_id"] == starts[1][1]  # 事件挂在 inner 上


def test_span_records_error_and_reraises():
    fake = _FakeExporter()
    telemetry._exporters = [fake]
    with pytest.raises(ValueError):
        with telemetry.span("boom"):
            raise ValueError("bad")
    end = [s for s in fake.spans if s[0] == "end"][0]
    assert end[3] == "bad"


def test_broken_exporter_never_breaks_flow():
    class _Broken:
        def emit(self, event):
            raise RuntimeError("exporter down")

        def start_span(self, *a):
            raise RuntimeError("down")

        def end_span(self, *a):
            raise RuntimeError("down")

    telemetry._exporters = [_Broken()]
    telemetry.emit_event(kind="tool_call", tool_name="t")  # 不得抛出
    with telemetry.span("still-ok"):
        pass
