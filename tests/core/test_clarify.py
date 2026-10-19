"""通用澄清节点：仅首轮触发、interrupt 载荷形态、resume 的选项 / 自由文本映射。"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from askanswer.config.schema import ContextSchema
from askanswer.core import clarify
from askanswer.core.state import SearchState
from askanswer.intents.base import ClarificationChoice, ClarificationRequest

_REQUEST = ClarificationRequest(
    prompt="选哪个？",
    choices=(
        ClarificationChoice(label="保持现状"),
        ClarificationChoice(label="切到 chat", updates={"intent": "chat"}),
    ),
    default_index=0,
    free_text_field="file_path",
    free_text_label="手动输入",
    free_text_prompt="路径：",
)


def test_interrupt_payload_shape():
    payload = clarify._interrupt_payload(_REQUEST)
    assert payload == {
        "type": "clarify",
        "prompt": "选哪个？",
        "labels": ["保持现状", "切到 chat"],
        "default_index": 0,
        "free_text": True,
        "free_text_label": "手动输入",
        "free_text_prompt": "路径：",
    }


@pytest.mark.parametrize(
    "decision,expected",
    [
        ({"index": 0}, {}),
        ({"index": 1}, {"intent": "chat"}),
        ({"index": 2, "text": " /tmp/a.txt "}, {"file_path": "/tmp/a.txt"}),
        ({"index": 2, "text": "   "}, {}),        # 自由文本留空 = 放弃
        ({"index": 9}, {}),                        # 越界 = 保持现状
        ({"index": "x"}, {}),                      # 非法 index
        (None, {}),                                # 非 dict（取消）
        ("1", {}),
    ],
)
def test_resolve_decision(decision, expected):
    assert clarify._resolve(_REQUEST, decision) == expected


class _Runtime:
    def __init__(self, context=None):
        self.context = context


def test_not_first_pass_is_noop():
    state = {"step": "retry_search", "intent": "sql"}
    assert clarify.clarify_node(state, _Runtime(ContextSchema())) == {}


def test_intent_without_clarify_is_noop():
    state = {"step": "understood", "intent": "math"}
    assert clarify.clarify_node(state, _Runtime(ContextSchema())) == {}


def test_sql_with_dsn_does_not_ask():
    state = {"step": "understood", "intent": "sql"}
    assert clarify.clarify_node(state, _Runtime(ContextSchema(db_dsn="sqlite://"))) == {}


def _clarify_graph():
    builder = StateGraph(SearchState, context_schema=ContextSchema)
    builder.add_node("clarify", clarify.clarify_node)
    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_sql_without_dsn_interrupts_and_resume_switches_intent():
    graph = _clarify_graph()
    config = {"configurable": {"thread_id": "c1"}}
    out = graph.invoke({"messages": [], "step": "understood", "intent": "sql"},
                       config=config, context=ContextSchema())
    (interrupt,) = out["__interrupt__"]
    assert interrupt.value["type"] == "clarify"
    assert "ASKANSWER_DB_DSN" in interrupt.value["prompt"]
    final = graph.invoke(Command(resume={"index": 1}), config=config, context=ContextSchema())
    assert final["intent"] == "chat"


def test_file_read_free_text_fills_path():
    graph = _clarify_graph()
    config = {"configurable": {"thread_id": "c2"}}
    out = graph.invoke({"messages": [], "step": "understood", "intent": "file_read",
                        "file_path": ""}, config=config, context=ContextSchema())
    payload = out["__interrupt__"][0].value
    assert payload["free_text"] is True
    final = graph.invoke(
        Command(resume={"index": len(payload["labels"]), "text": "./notes.md"}),
        config=config, context=ContextSchema(),
    )
    assert final["file_path"] == "./notes.md"


def test_failing_clarify_degrades_to_noop(monkeypatch):
    from askanswer.intents import get_intent_registry

    handler = get_intent_registry().get("sql")

    def _boom(state, context):
        raise RuntimeError("bad handler")

    monkeypatch.setattr(handler, "clarify", _boom, raising=False)
    state = {"step": "understood", "intent": "sql"}
    assert clarify.clarify_node(state, _Runtime(ContextSchema())) == {}
