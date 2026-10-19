"""角色路由：默认跟随全局代理、env 配置主模型与回退链、变换重放、流式回退守卫、审计归因。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from askanswer.llm import proxy, routing
from askanswer.observability import audit
from askanswer.storage import NULL_PERSISTENCE


class _Backend:
    """可配置失败点的假 backend；记录 bind_tools / with_structured_output 变换。"""

    def __init__(self, label, *, fail=False, chunks=("a", "b"), fail_after_first=False):
        self.label = label
        self.fail = fail
        self.chunks = chunks
        self.fail_after_first = fail_after_first
        self.transforms = []
        self.invocations = []

    def bind_tools(self, tools, **kwargs):
        self.transforms.append(("bind_tools", tuple(tools)))
        return self

    def with_structured_output(self, schema, **kwargs):
        self.transforms.append(("with_structured_output", schema))
        return self

    def invoke(self, messages, config=None, **kwargs):
        self.invocations.append(config)
        if self.fail:
            raise RuntimeError(f"{self.label} down")
        return AIMessage(content=f"from {self.label}")

    def stream(self, messages, config=None, **kwargs):
        if self.fail:
            raise RuntimeError(f"{self.label} down")
        for i, chunk in enumerate(self.chunks):
            if self.fail_after_first and i == 1:
                raise RuntimeError("mid-stream failure")
            yield chunk


@pytest.fixture
def backends(monkeypatch):
    """按 label 返回预置的假 backend；清理路由表与 backend 缓存。"""
    table: dict[str, _Backend] = {}

    def _build(label, **kwargs):
        if label not in table:
            raise ImportError(f"no SDK for {label}")
        return table[label]

    monkeypatch.setattr(routing, "build_backend", _build)
    routing.reset_router()
    routing._BACKEND_CACHE.clear()
    yield table
    routing.reset_router()
    routing._BACKEND_CACHE.clear()


@pytest.fixture
def audit_run():
    tokens = audit.begin_run("route-thread", persistence=NULL_PERSISTENCE)
    yield
    audit.end_run(tokens)


def _pending_kinds():
    return [e["kind"] for e in (audit._PENDING.get() or [])]


def test_normalize_spec():
    assert routing.normalize_spec("gpt-4o") == "openai:gpt-4o"
    assert routing.normalize_spec(" anthropic:claude ") == "anthropic:claude"


def test_default_route_returns_global_proxy(backends, monkeypatch):
    monkeypatch.delenv("ASKANSWER_MODEL_ANSWER", raising=False)
    monkeypatch.delenv("ASKANSWER_MODEL_FALLBACKS_ANSWER", raising=False)
    assert routing.model_for(routing.ROLE_ANSWER) is proxy.model


def test_env_route_builds_routed_model(backends, monkeypatch):
    monkeypatch.setenv("ASKANSWER_MODEL_CLASSIFY", "mini")
    backends["openai:mini"] = _Backend("mini")
    llm = routing.model_for(routing.ROLE_CLASSIFY)
    assert isinstance(llm, routing.RoutedModel)
    assert llm.label == "openai:mini"
    assert llm.invoke("hi").content == "from mini"


def test_invoke_falls_back_and_logs(backends, monkeypatch, audit_run):
    monkeypatch.setenv("ASKANSWER_MODEL_ANSWER", "openai:primary")
    monkeypatch.setenv("ASKANSWER_MODEL_FALLBACKS_ANSWER", "openai:missing, deepseek:backup")
    backends["openai:primary"] = _Backend("primary", fail=True)
    backends["deepseek:backup"] = _Backend("backup")
    result = routing.model_for(routing.ROLE_ANSWER).invoke("q")
    assert result.content == "from backup"
    fallbacks = [e for e in audit._PENDING.get() if e["kind"] == "model_fallback"]
    # 主模型调用失败 + 回退候选 SDK 缺失（构建失败）各记一条
    assert [e["model_label"] for e in fallbacks] == ["openai:primary", "openai:missing"]
    assert all(e["args_summary"] == "role=answer" for e in fallbacks)


def test_audit_callback_attributed_to_executing_model(backends, monkeypatch):
    monkeypatch.setenv("ASKANSWER_MODEL_EVALUATE", "openai:judge")
    backend = _Backend("judge")
    backends["openai:judge"] = backend
    routing.model_for(routing.ROLE_EVALUATE).invoke("q")
    callbacks = backend.invocations[-1]["callbacks"]
    labels = [getattr(cb, "model_label", None) for cb in callbacks]
    assert "openai:judge" in labels


def test_transforms_replayed_on_every_candidate(backends, monkeypatch):
    monkeypatch.setenv("ASKANSWER_MODEL_ANSWER", "openai:a")
    monkeypatch.setenv("ASKANSWER_MODEL_FALLBACKS_ANSWER", "openai:b")
    a = backends["openai:a"] = _Backend("a", fail=True)
    b = backends["openai:b"] = _Backend("b")
    bound = routing.model_for(routing.ROLE_ANSWER).bind_tools(["t1"]).with_structured_output(dict)
    assert bound.invoke("q").content == "from b"
    expected = [("bind_tools", ("t1",)), ("with_structured_output", dict)]
    assert a.transforms == expected and b.transforms == expected


def test_all_candidates_fail_raises_last_error(backends, monkeypatch):
    monkeypatch.setenv("ASKANSWER_MODEL_ANSWER", "openai:a")
    monkeypatch.setenv("ASKANSWER_MODEL_FALLBACKS_ANSWER", "openai:b")
    backends["openai:a"] = _Backend("a", fail=True)
    backends["openai:b"] = _Backend("b", fail=True)
    with pytest.raises(RuntimeError, match="b down"):
        routing.model_for(routing.ROLE_ANSWER).invoke("q")


def test_stream_falls_back_before_first_chunk(backends, monkeypatch):
    monkeypatch.setenv("ASKANSWER_MODEL_ANSWER", "openai:a")
    monkeypatch.setenv("ASKANSWER_MODEL_FALLBACKS_ANSWER", "openai:b")
    backends["openai:a"] = _Backend("a", fail=True)
    backends["openai:b"] = _Backend("b", chunks=("x", "y"))
    assert list(routing.model_for(routing.ROLE_ANSWER).stream("q")) == ["x", "y"]


def test_stream_does_not_fall_back_after_first_chunk(backends, monkeypatch):
    """首个分片已产出后失败不得回退（否则会向用户重复输出）。"""
    monkeypatch.setenv("ASKANSWER_MODEL_ANSWER", "openai:a")
    monkeypatch.setenv("ASKANSWER_MODEL_FALLBACKS_ANSWER", "openai:b")
    backends["openai:a"] = _Backend("a", chunks=("x", "y"), fail_after_first=True)
    b = backends["openai:b"] = _Backend("b")
    stream = routing.model_for(routing.ROLE_ANSWER).stream("q")
    assert next(stream) == "x"
    with pytest.raises(RuntimeError, match="mid-stream"):
        next(stream)
    assert b.invocations == []


def test_backend_cache_reused(backends, monkeypatch):
    monkeypatch.setenv("ASKANSWER_MODEL_SUMMARIZE", "openai:s")
    built = []

    def _build(label, **kwargs):
        built.append(label)
        return _Backend(label)

    monkeypatch.setattr(routing, "build_backend", _build)
    llm = routing.model_for(routing.ROLE_SUMMARIZE)
    llm.invoke("1")
    llm.invoke("2")
    assert built == ["openai:s"]


def test_describe_routes(backends, monkeypatch):
    monkeypatch.setenv("ASKANSWER_MODEL_CLASSIFY", "mini")
    monkeypatch.setenv("ASKANSWER_MODEL_FALLBACKS_CLASSIFY", "anthropic:haiku")
    table = routing.describe_routes()
    assert table["classify"] == "openai:mini → anthropic:haiku"
    assert table["answer"].startswith("(全局 ")
