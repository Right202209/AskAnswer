"""P9 回归：HITL 恢复后，普通工具与已批准的确认工具都只执行一次。

旧实现在同一个 tools 节点里先执行普通工具再 interrupt()，LangGraph 恢复时
节点从头重跑 → 普通工具执行两次；多个确认项时前面已批准的动作也会被重放。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from askanswer import confirmations
from askanswer.react import build_react_subgraph
from askanswer.registry import ToolDescriptor, ToolRegistry
from askanswer.runtime import build_container
from askanswer.schema import ContextSchema
from askanswer.state import SearchState

CALLS: dict[str, int] = {}


@tool
def plain_counter(x: str) -> str:
    """普通工具：记录被执行的次数。"""
    CALLS["plain"] = CALLS.get("plain", 0) + 1
    return f"plain:{x}"


@tool
def risky_op(x: str) -> str:
    """需确认的工具（由假的确认类执行）。"""
    return x


class _CountingConfirmation:
    clazz = "fake_confirm"

    def plan(self, tool_call):
        return {"x": tool_call["args"]["x"]}

    def gate(self, payload):
        return "blocked!" if payload.get("x") == "bad" else None

    def interrupt_payload(self, payload):
        return {"type": "confirm_fake", "x": payload["x"]}

    def audit_args(self, payload):
        return {"x": payload.get("x")}

    def apply(self, payload, decision, tool_call):
        key = f"apply:{payload['x']}"
        if decision is True:
            CALLS[key] = CALLS.get(key, 0) + 1
            return confirmations.ConfirmationOutcome(content=f"done {payload['x']}", approved=True)
        return confirmations.ConfirmationOutcome(content="cancelled", approved=False)


class _ScriptedModel:
    """第一轮返回混合 tool_calls，之后返回最终答案。"""

    label = "fake:model"

    def __init__(self, tool_calls):
        self._tool_calls = tool_calls
        self.turn = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.turn += 1
        if self.turn == 1:
            return AIMessage(content="", tool_calls=self._tool_calls)
        return AIMessage(content="final")


def _setup(monkeypatch, tool_calls):
    """依赖全部通过 Container 注入（P2）：不再 monkeypatch 任何模块全局。"""
    CALLS.clear()
    reg = ToolRegistry()
    reg.register(ToolDescriptor(tool=plain_counter, tags=frozenset({"chat"}), source="builtin"))
    reg.register(
        ToolDescriptor(
            tool=risky_op, tags=frozenset({"chat"}), source="builtin",
            confirmation_class="fake_confirm",
        )
    )
    handler = _CountingConfirmation()
    model = _ScriptedModel(tool_calls)
    container = build_container(
        tools=reg,
        models=lambda role: model,
        confirmations=lambda clazz: handler if clazz == "fake_confirm" else None,
    )

    builder = StateGraph(SearchState, context_schema=ContextSchema)
    builder.add_node("answer", build_react_subgraph(container))
    builder.add_edge(START, "answer")
    app = builder.compile(checkpointer=InMemorySaver())
    return app


def _start(app, thread):
    cfg = {"configurable": {"thread_id": thread}}
    state = {
        "messages": [HumanMessage(content="do it")],
        "intent": "chat",
        "step": "tool_called_init",
        "user_query": "do it",
    }
    app.invoke(state, config=cfg, context=ContextSchema())
    return cfg


def _interrupt_value(app, cfg):
    snap = app.get_state(cfg)
    for task in snap.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def test_plain_tool_runs_once_after_resume(monkeypatch):
    app = _setup(monkeypatch, [
        {"name": "plain_counter", "args": {"x": "a"}, "id": "c1"},
        {"name": "risky_op", "args": {"x": "r1"}, "id": "c2"},
    ])
    cfg = _start(app, "t1")
    assert _interrupt_value(app, cfg) == {"type": "confirm_fake", "x": "r1"}
    # interrupt 之前不能执行任何工具
    assert CALLS == {}
    app.invoke(Command(resume=True), config=cfg, context=ContextSchema())
    assert CALLS == {"plain": 1, "apply:r1": 1}
    assert app.get_state(cfg).values["final_answer"] == "final"


def test_two_confirmations_each_applied_once(monkeypatch):
    app = _setup(monkeypatch, [
        {"name": "risky_op", "args": {"x": "r1"}, "id": "c1"},
        {"name": "plain_counter", "args": {"x": "a"}, "id": "c2"},
        {"name": "risky_op", "args": {"x": "r2"}, "id": "c3"},
    ])
    cfg = _start(app, "t2")
    assert _interrupt_value(app, cfg)["x"] == "r1"
    app.invoke(Command(resume=True), config=cfg, context=ContextSchema())
    assert _interrupt_value(app, cfg)["x"] == "r2"
    assert CALLS == {}  # 第二个确认项决定之前什么都不执行
    app.invoke(Command(resume=False), config=cfg, context=ContextSchema())
    assert CALLS == {"plain": 1, "apply:r1": 1}


def test_blocked_item_never_interrupts(monkeypatch):
    app = _setup(monkeypatch, [
        {"name": "risky_op", "args": {"x": "bad"}, "id": "c1"},
        {"name": "plain_counter", "args": {"x": "a"}, "id": "c2"},
    ])
    cfg = _start(app, "t3")
    assert _interrupt_value(app, cfg) is None
    assert CALLS == {"plain": 1}
    tool_msgs = [m for m in app.get_state(cfg).values["messages"] if m.type == "tool"]
    assert any(m.content == "blocked!" for m in tool_msgs)
