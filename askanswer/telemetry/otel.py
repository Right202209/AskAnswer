"""OpenTelemetry exporter：把 telemetry span/事件映射成 OTEL span。

仅在 ``opentelemetry`` SDK 可导入且 ``ASKANSWER_OTEL_EXPORTER`` 存在时构建。
本模块只负责“产出 span”，具体导出到哪（OTLP/console/…）由宿主的 OTEL
provider 配置决定 —— 与 AskAnswer 解耦，符合“标准 exporter”定位。

限制（第一版）：span 按 span_id 存取、扁平记录，不做 OTEL context 激活，
因此父子 span 不会在 OTEL 侧自动嵌套；如需严格 trace 树可后续接入
``opentelemetry.context``。
"""

from __future__ import annotations

from typing import Any


def build_otel_exporter():
    """构建 OTEL exporter；SDK 缺失时返回 None（宿主未装 opentelemetry 即视为未启用）。"""
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    tracer = trace.get_tracer("askanswer")
    return _OtelExporter(tracer)


class _OtelExporter:
    """span_id → OTEL span 的薄适配层。"""

    def __init__(self, tracer):
        self._tracer = tracer
        self._spans: dict[str, Any] = {}

    def start_span(self, span_id: str, name: str, parent_id: str | None, attrs: dict) -> None:
        try:
            otel_span = self._tracer.start_span(name)
        except Exception:
            return
        for key, value in (attrs or {}).items():
            try:
                otel_span.set_attribute(str(key), _attr_value(value))
            except Exception:
                pass
        self._spans[span_id] = otel_span

    def end_span(self, span_id: str, duration_ms: int, error: str | None) -> None:
        otel_span = self._spans.pop(span_id, None)
        if otel_span is None:
            return
        try:
            otel_span.set_attribute("duration_ms", int(duration_ms))
            if error:
                otel_span.set_attribute("error", error)
            otel_span.end()
        except Exception:
            pass

    def emit(self, event: dict[str, Any]) -> None:
        span_id = event.get("span_id")
        otel_span = self._spans.get(span_id) if span_id else None
        if otel_span is None:
            return
        try:
            otel_span.add_event(
                event.get("kind") or "event",
                attributes=_event_attributes(event),
            )
        except Exception:
            pass


def _attr_value(value: Any):
    """OTEL 属性只接受 str/bool/数字及其序列；其它类型转成字符串。"""
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _event_attributes(event: dict[str, Any]) -> dict:
    """把事件里的非空标量字段整理成 OTEL event 属性。"""
    keys = ("tool_name", "duration_ms", "input_tokens", "output_tokens", "model_label", "error")
    return {key: _attr_value(event[key]) for key in keys if event.get(key) is not None}
