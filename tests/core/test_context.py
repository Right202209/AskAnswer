"""上下文预算：零开销直通、保留 system 前缀与最新块、工具调用原子块、摘要模式与 env 解析。"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from askanswer.core import context as ctx


def _tool_block(call_id: str, text: str = "x" * 400):
    ai = AIMessage(content="", tool_calls=[{"name": "calc", "args": {"e": "1+1"}, "id": call_id}])
    return [ai, ToolMessage(content=text, tool_call_id=call_id, name="calc")]


# ── token 估算 ───────────────────────────────────────────────────────────

def test_estimate_tokens_empty_and_cjk_heavier():
    assert ctx.estimate_tokens("") == 0
    ascii_cost = ctx.estimate_tokens("a" * 60)
    cjk_cost = ctx.estimate_tokens("中" * 60)
    assert cjk_cost > ascii_cost  # CJK 每 token 字符更少，估算更保守


def test_message_tokens_counts_tool_calls():
    plain = AIMessage(content="hi")
    with_call = AIMessage(content="hi", tool_calls=[{"name": "t", "args": {"q": "x" * 80}, "id": "1"}])
    assert ctx.message_tokens(with_call) > ctx.message_tokens(plain)


def test_list_content_blocks_are_counted():
    msg = HumanMessage(content=[{"type": "text", "text": "hello world"}, "tail"])
    assert ctx._content_text(msg) == "hello world tail"


# ── 预算裁剪 ─────────────────────────────────────────────────────────────

def test_no_budget_is_zero_cost_passthrough(monkeypatch):
    monkeypatch.setattr(ctx, "message_tokens", lambda m: pytest.fail("不应估算 token"))
    messages = [HumanMessage(content="a"), AIMessage(content="b")]
    result = ctx.budget_messages(messages, max_tokens=None)
    assert result.messages is messages
    assert result.dropped == 0 and result.digest_text == ""


def test_keeps_system_prefix_and_newest_even_when_over_budget():
    system = SystemMessage(content="rules")
    history = [HumanMessage(content=f"old {i} " + "x" * 200) for i in range(5)]
    newest = HumanMessage(content="newest " + "y" * 2000)
    result = ctx.budget_messages([system, *history, newest], max_tokens=50)
    assert result.messages[0] is system
    assert result.messages[-1] is newest
    assert result.dropped == 5


def test_drops_oldest_first_within_budget():
    msgs = [HumanMessage(content=f"m{i} " + "z" * 100) for i in range(6)]
    per = ctx.message_tokens(msgs[0])
    result = ctx.budget_messages(msgs, max_tokens=per * 3)
    assert result.messages == msgs[-3:]
    assert result.dropped == 3
    assert result.kept_tokens <= per * 3


def test_tool_call_block_is_atomic():
    """AIMessage(tool_calls) 与其 ToolMessage 同进同出，绝不产生孤儿 ToolMessage。"""
    first = _tool_block("c1")
    second = _tool_block("c2")
    newest = HumanMessage(content="now")
    messages = [HumanMessage(content="q"), *first, *second, newest]
    budget = sum(ctx.message_tokens(m) for m in [*second, newest]) + 1
    result = ctx.budget_messages(messages, max_tokens=budget)
    kept = result.messages
    assert kept == [*second, newest]
    for i, msg in enumerate(kept):
        if isinstance(msg, ToolMessage):
            assert getattr(kept[i - 1], "tool_calls", None) or isinstance(kept[i - 1], ToolMessage)


def test_group_blocks_groups_multiple_tool_messages():
    ai = AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "1"},
                                            {"name": "b", "args": {}, "id": "2"}])
    t1 = ToolMessage(content="r1", tool_call_id="1")
    t2 = ToolMessage(content="r2", tool_call_id="2")
    blocks = ctx._group_blocks([HumanMessage(content="q"), ai, t1, t2])
    assert [len(b) for b in blocks] == [1, 3]


# ── 摘要模式 ─────────────────────────────────────────────────────────────

def _overflowing():
    old = [HumanMessage(content=f"旧问题 {i}") for i in range(4)]
    newest = HumanMessage(content="新问题 " + "n" * 400)
    return [*old, newest], ctx.message_tokens(newest)


def test_brief_digest_lists_dropped_history():
    messages, budget = _overflowing()
    result = ctx.budget_messages(messages, max_tokens=budget, digest=ctx.DIGEST_BRIEF)
    assert result.digest_text.startswith("[已省略 4 条更早的历史")
    assert "human: 旧问题 0" in result.digest_text
    assert len(result.dropped_labels) == 4


def test_digest_off_has_no_text():
    messages, budget = _overflowing()
    assert ctx.budget_messages(messages, max_tokens=budget).digest_text == ""


def test_llm_digest_falls_back_to_brief_on_failure(monkeypatch):
    from askanswer.llm import routing

    def _boom(role):
        raise RuntimeError("no model")

    monkeypatch.setattr(routing, "model_for", _boom)
    messages, budget = _overflowing()
    result = ctx.budget_messages(messages, max_tokens=budget, digest=ctx.DIGEST_LLM)
    assert result.digest_text.startswith("[已省略 4 条更早的历史，要点如下]")


def test_llm_digest_uses_summarize_route(monkeypatch):
    from askanswer.llm import routing

    roles = []

    class _Summarizer:
        def invoke(self, messages):
            return AIMessage(content="- 要点一\n- 要点二")

    def _model_for(role):
        roles.append(role)
        return _Summarizer()

    monkeypatch.setattr(routing, "model_for", _model_for)
    messages, budget = _overflowing()
    result = ctx.budget_messages(messages, max_tokens=budget, digest=ctx.DIGEST_LLM)
    assert roles == [routing.ROLE_SUMMARIZE]
    assert "摘要如下" in result.digest_text and "要点二" in result.digest_text


# ── env 解析 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [("", None), ("abc", None), ("0", None), ("-5", None),
                                          ("24000", 24000)])
def test_answer_token_budget(monkeypatch, raw, expected):
    monkeypatch.setenv("ASKANSWER_CONTEXT_MAX_TOKENS", raw)
    assert ctx.answer_token_budget() == expected


@pytest.mark.parametrize("raw,expected", [("", "off"), ("brief", "brief"), ("1", "brief"),
                                          ("LLM", "llm"), ("weird", "off")])
def test_digest_mode(monkeypatch, raw, expected):
    monkeypatch.setenv("ASKANSWER_CONTEXT_DIGEST", raw)
    assert ctx.digest_mode() == expected
