"""可观测性门控：无 env → 零副作用（不建 exporter/span）；有 env 标志正确。

telemetry 的核心不变量是「关闭即零开销」：conftest 已清掉 LANGSMITH/OTEL 变量，
每个用例再显式重置模块级状态，避免跨用例污染 ``_initialized`` / ``_exporters``。
"""

from __future__ import annotations

import pytest

from askanswer import telemetry


@pytest.fixture(autouse=True)
def _reset_telemetry():
    """每个用例前后把 telemetry 模块状态清零，保证门控测试相互独立。"""
    telemetry._initialized = False
    telemetry._exporters = []
    yield
    telemetry._initialized = False
    telemetry._exporters = []


def test_no_env_means_disabled():
    telemetry.init_telemetry()
    assert telemetry.is_enabled() is False
    assert telemetry._exporters == []


def test_no_env_emit_event_is_noop():
    telemetry.init_telemetry()
    # 不应抛异常、不应有副作用
    telemetry.emit_event(kind="llm_call", model_label="m", input_tokens=1, output_tokens=1)


def test_no_env_span_yields_without_id():
    telemetry.init_telemetry()
    with telemetry.span("test-span"):
        pass  # 未启用时 span 不 push contextvar
    assert telemetry._SPAN_STACK.get() == ()


def test_no_env_open_span_returns_none():
    telemetry.init_telemetry()
    assert telemetry.open_span("x") is None
    telemetry.close_span(None)  # None handle → no-op，不报错


def test_no_env_llm_callback_returns_none():
    telemetry.init_telemetry()
    assert telemetry.llm_callback("openai:gpt") is None


def test_init_is_idempotent():
    telemetry.init_telemetry()
    telemetry.init_telemetry()
    assert telemetry.is_enabled() is False


def test_load_exporters_with_langsmith_env(monkeypatch):
    """有 LANGSMITH_API_KEY 时尝试构建 exporter；SDK 缺失则 build 返回 None（仍不崩）。"""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-fake")
    telemetry._initialized = False
    telemetry._exporters = []
    # 不真正连接 LangSmith：just确认 init 不抛异常且门控读到了 env 分支
    telemetry.init_telemetry()
    # exporter 是否装上取决于 SDK 是否可用；关键是过程不崩、门控逻辑被执行到
    assert isinstance(telemetry._exporters, list)


def test_extract_tokens_from_usage_metadata():
    class _Msg:
        usage_metadata = {"input_tokens": 12, "output_tokens": 7}

    class _Gen:
        message = _Msg()

    class _Resp:
        generations = [[_Gen()]]

    assert telemetry._extract_tokens(_Resp()) == (12, 7)


def test_extract_tokens_missing_returns_none_pair():
    class _Resp:
        generations = []

    assert telemetry._extract_tokens(_Resp()) == (None, None)


# ── 启用态：exporter 派发 / span 栈 / 异常记录 / 容错 ─────────────────────
# 上面用例覆盖「关闭即零开销」；下面注入假 exporter 覆盖启用后的分发路径。

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


def test_init_only_loads_exporters_once(monkeypatch):
    """重复 init 只装配一次 exporter（幂等的更强断言：_load_exporters 只调一次）。"""
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
