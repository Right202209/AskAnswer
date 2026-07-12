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
