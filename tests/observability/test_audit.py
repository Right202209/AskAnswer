"""审计：contextvar 生命周期、排队 / 立即写入、成本闸门读数、usage 抽取（含 cached_tokens）与落库。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from askanswer.observability import audit


class _Recorder:
    def __init__(self):
        self.batch = []
        self.single = []

    def log_audit_events(self, events):
        self.batch.extend(dict(e) for e in events)
        return len(events)

    def log_audit_event(self, thread_id, **fields):
        self.single.append({"thread_id": thread_id, **fields})

    def upsert_meta(self, thread_id, **fields):
        pass


# ── run 生命周期 ─────────────────────────────────────────────────────────

def test_begin_end_run_restores_context(monkeypatch):
    monkeypatch.setenv("ASKANSWER_TENANT_ID", "env-tenant")
    outer = audit.begin_run("outer", persistence=_Recorder())
    assert audit.current_thread_id() == "outer"
    assert audit.current_tenant_id() == "env-tenant"  # 未显式给时回退 env
    inner = audit.begin_run("inner", tenant_id="explicit", persistence=_Recorder())
    assert (audit.current_thread_id(), audit.current_tenant_id()) == ("inner", "explicit")
    audit.end_run(inner)
    assert (audit.current_thread_id(), audit.current_tenant_id()) == ("outer", "env-tenant")
    audit.end_run(outer)
    assert audit._PENDING.get() is None


def test_events_queue_during_run_and_flush_to_bound_backend():
    backend = _Recorder()
    tokens = audit.begin_run("t1", tenant_id="acme", persistence=backend)
    try:
        audit.log_event(kind="tool_call", tool_name="calc")
        audit.log_event(kind="llm_call", model_label="m", input_tokens=3, output_tokens=1)
        assert backend.batch == []  # 先缓冲
        assert audit.flush_pending(thread_id="t1", intent="math") == 2
    finally:
        audit.end_run(tokens)
    assert {e["kind"] for e in backend.batch} == {"tool_call", "llm_call"}
    assert all(e["tenant_id"] == "acme" and e["intent"] == "math" for e in backend.batch)


def test_immediate_event_goes_to_bound_backend():
    backend = _Recorder()
    tokens = audit.begin_run("t2", persistence=backend)
    try:
        audit.log_event(kind="mcp_connect", tool_name="srv", immediate=True)
    finally:
        audit.end_run(tokens)
    assert backend.single and backend.single[0]["kind"] == "mcp_connect"


def test_flush_failure_is_swallowed():
    class _Broken(_Recorder):
        def log_audit_events(self, events):
            raise RuntimeError("disk full")

    tokens = audit.begin_run("t3", persistence=_Broken())
    try:
        audit.log_event(kind="tool_call")
        assert audit.flush_pending() == 0
        assert audit._PENDING.get() == []
    finally:
        audit.end_run(tokens)


# ── 成本闸门读数 ─────────────────────────────────────────────────────────

def test_run_usage_so_far_counts_llm_calls_only():
    tokens = audit.begin_run("t4", persistence=_Recorder())
    try:
        audit.log_event(kind="llm_call", input_tokens=100, output_tokens=20)
        audit.log_event(kind="llm_call", input_tokens=None, output_tokens=5)
        audit.log_event(kind="tool_call", input_tokens=999)
        assert audit.run_usage_so_far() == (100, 25)
    finally:
        audit.end_run(tokens)
    assert audit.run_usage_so_far() == (0, 0)


def test_sorcery_budget_gate(monkeypatch):
    from askanswer.core import nodes

    tokens = audit.begin_run("t5", persistence=_Recorder())
    try:
        audit.log_event(kind="llm_call", input_tokens=60, output_tokens=50)
        monkeypatch.delenv("ASKANSWER_RUN_TOKEN_BUDGET", raising=False)
        assert nodes._run_budget_stop_reason() is None
        monkeypatch.setenv("ASKANSWER_RUN_TOKEN_BUDGET", "1000")
        assert nodes._run_budget_stop_reason() is None
        monkeypatch.setenv("ASKANSWER_RUN_TOKEN_BUDGET", "100")
        assert "110" in nodes._run_budget_stop_reason()
    finally:
        audit.end_run(tokens)


# ── usage 抽取（含缓存命中） ─────────────────────────────────────────────

def _response_with_message_usage(usage: dict):
    message = SimpleNamespace(usage_metadata=usage, response_metadata={})
    return SimpleNamespace(llm_output=None, generations=[[SimpleNamespace(message=message)]])


@pytest.mark.parametrize(
    "response,expected",
    [
        (SimpleNamespace(llm_output={"token_usage": {
            "prompt_tokens": 120, "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 64}}}, generations=[]),
         (120, 30, 64)),
        (_response_with_message_usage({"input_tokens": 50, "output_tokens": 5,
                                       "input_token_details": {"cache_read": 40}}),
         (50, 5, 40)),
        (SimpleNamespace(llm_output={"usage": {"input_tokens": 9, "output_tokens": 1,
                                               "cache_read_input_tokens": 8}}, generations=[]),
         (9, 1, 8)),
        (_response_with_message_usage({"input_tokens": 7, "output_tokens": 2}), (7, 2, None)),
        (SimpleNamespace(llm_output=None, generations=[]), (None, None, None)),
    ],
)
def test_extract_usage_variants(response, expected):
    assert audit._extract_usage(response) == expected


def test_llm_callback_records_cached_tokens_and_persists(pm):
    tokens = audit.begin_run("cache-thread", persistence=pm)
    try:
        callback = audit.LLMUsageCallback(model_label="openai:gpt")
        callback.on_llm_start()
        callback.on_llm_end(_response_with_message_usage(
            {"input_tokens": 200, "output_tokens": 10, "input_token_details": {"cache_read": 128}}
        ))
        (event,) = audit._PENDING.get()
        assert event["cached_tokens"] == 128
        audit.flush_pending(thread_id="cache-thread")
    finally:
        audit.end_run(tokens)
    (stored,) = pm.list_audit_events(thread_id="cache-thread")
    assert (stored.input_tokens, stored.output_tokens, stored.cached_tokens) == (200, 10, 128)
    assert pm.usage_summary(thread_id="cache-thread")["models"][0]["cached_tokens"] == 128


def test_llm_callback_error_path():
    tokens = audit.begin_run("err-thread", persistence=_Recorder())
    try:
        audit.LLMUsageCallback(model_label="m").on_llm_error(RuntimeError("429"))
        (event,) = audit._PENDING.get()
        assert event["error"] == "429" and event["kind"] == "llm_call"
    finally:
        audit.end_run(tokens)


def test_with_llm_audit_callback_appends_without_mutating():
    original = {"callbacks": ["existing"], "tags": ["x"]}
    cfg = audit.with_llm_audit_callback(original, model_label="m")
    assert original["callbacks"] == ["existing"]
    assert cfg["callbacks"][0] == "existing"
    assert isinstance(cfg["callbacks"][1], audit.LLMUsageCallback)


def test_summarize_args_is_bounded():
    text = audit.summarize_args({"k": "v" * 500})
    assert len(text) == 200 and text.endswith("…")
