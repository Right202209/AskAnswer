"""OTEL exporter：子 span 以父 span 的 context 启动（嵌套 trace 树），而非扁平记录。"""

from __future__ import annotations

import pytest

from askanswer.observability import telemetry
from askanswer.observability.otel import _OtelExporter

trace = pytest.importorskip("opentelemetry.trace")


class _RecordingTracer:
    """记录每次 start_span 的 (name, parent span) —— parent 从传入的 context 里解析。"""

    def __init__(self):
        self.started: list[tuple[str, object]] = []

    def start_span(self, name, context=None):
        parent = trace.get_current_span(context) if context is not None else None
        span = trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT)
        self.started.append((name, parent))
        return span


@pytest.fixture
def exporter():
    tracer = _RecordingTracer()
    exp = _OtelExporter(tracer, trace.set_span_in_context)
    telemetry._initialized = True
    telemetry._exporters = [exp]
    yield exp, tracer
    telemetry._initialized = False
    telemetry._exporters = []


def test_root_span_has_no_parent_context(exporter):
    exp, tracer = exporter
    exp.start_span("root", "askanswer.query", None, {})
    assert tracer.started == [("askanswer.query", None)]


def test_child_span_started_in_parent_context(exporter):
    exp, tracer = exporter
    exp.start_span("p", "parent", None, {})
    parent_span = exp._spans["p"]
    exp.start_span("c", "child", "p", {})
    assert tracer.started[1][0] == "child"
    assert tracer.started[1][1] is parent_span


def test_nested_spans_through_telemetry_facade(exporter):
    _, tracer = exporter
    with telemetry.span("outer"):
        with telemetry.span("inner"):
            pass
    (outer_name, outer_parent), (inner_name, inner_parent) = tracer.started
    assert (outer_name, outer_parent) == ("outer", None)
    assert inner_name == "inner" and inner_parent is not None
    assert telemetry._SPAN_STACK.get() == ()


def test_unknown_parent_falls_back_to_root(exporter):
    exp, tracer = exporter
    exp.start_span("x", "orphan", "missing-parent", {})
    assert tracer.started == [("orphan", None)]
