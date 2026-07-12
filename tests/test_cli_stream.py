"""runner 事件流契约 + CLI ``stream_query`` 消费口径（C1 拆分后：CLI 复用 runner，不再各造一份）。

这些用例锁住 ``runner.stream_leg`` 的事件顺序与 ``cli.stream.stream_query`` 的消费映射，
是 C1 把 stream_query 收敛到 runner 之上后的回归网（server 也依赖同一契约）。
"""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk

from askanswer import runner
from askanswer.cli import stream as cli_stream


class _FakeState:
    def __init__(self, values: dict, tasks=()):
        self.values = values
        self.tasks = tasks


class _FakeApp:
    """按“腿”回放脚本：每次 ``stream`` 调用弹出一段 ``(chunk_mode, payload)`` 序列。"""

    def __init__(self, legs: list[list[tuple[str, object]]], state_values: dict):
        self._legs = list(legs)
        self._state_values = state_values
        self.stream_calls: list[object] = []

    def stream(self, graph_input, *, config, context, stream_mode):
        self.stream_calls.append(graph_input)
        leg = self._legs.pop(0) if self._legs else []
        yield from leg

    def get_state(self, config):
        return _FakeState(self._state_values)


def _token(text: str) -> tuple[str, tuple]:
    return ("messages", (AIMessageChunk(content=text), {}))


def _tool_plan() -> tuple[str, tuple]:
    chunk = AIMessageChunk(
        content="", tool_call_chunks=[{"name": "search", "args": "", "id": "1", "index": 0}]
    )
    return ("messages", (chunk, {}))


def _node(name: str, update: dict) -> tuple[str, dict]:
    return ("updates", {name: update})


def _interrupt(payload: dict) -> tuple[str, dict]:
    return ("updates", {"__interrupt__": payload})


# ── runner.stream_leg：事件契约 ───────────────────────────────────────


def test_stream_leg_emits_token_tool_node_then_final():
    app = _FakeApp(
        legs=[[
            _node("understand", {"intent": "chat"}),
            _tool_plan(),
            _token("Hello "),
            _token("world"),
            _node("answer", {"final_answer": "Hello world"}),
        ]],
        state_values={"final_answer": "Hello world", "messages": []},
    )
    events = list(runner.stream_leg(
        app, runner.query_input("hi"),
        config=runner.thread_config("t1"), context=runner.runtime_context_from_env(),
    ))
    kinds = [e.kind for e in events]
    # 一个规划阶段只发一次 tool 事件；正文 token 逐段发；最后以 final 收尾。
    assert kinds == [
        runner.EVENT_NODE, runner.EVENT_TOOL, runner.EVENT_TOKEN,
        runner.EVENT_TOKEN, runner.EVENT_NODE, runner.EVENT_FINAL,
    ]
    assert [e.text for e in events if e.kind == runner.EVENT_TOKEN] == ["Hello ", "world"]
    assert events[-1].text == "Hello world"


def test_stream_leg_interrupt_yields_interrupt_and_no_final():
    payload = {"type": "confirm_shell", "command": "ls"}
    app = _FakeApp(
        legs=[[_node("shell_plan", {"pending_shell": {"1": {}}}), _interrupt(payload)]],
        state_values={},
    )
    events = list(runner.stream_leg(
        app, runner.query_input("run ls"),
        config=runner.thread_config("t1"), context=runner.runtime_context_from_env(),
    ))
    assert events[-1].kind == runner.EVENT_INTERRUPT
    assert events[-1].data == payload
    assert not any(e.kind == runner.EVENT_FINAL for e in events)


def test_stream_leg_final_answer_falls_back_to_state():
    # 节点流未给出 final_answer 时，final 事件从 state 兜底取。
    app = _FakeApp(
        legs=[[_node("understand", {"intent": "chat"})]],
        state_values={"final_answer": "from state"},
    )
    events = list(runner.stream_leg(
        app, runner.query_input("hi"),
        config=runner.thread_config("t1"), context=runner.runtime_context_from_env(),
    ))
    assert events[-1].kind == runner.EVENT_FINAL
    assert events[-1].text == "from state"


# ── cli.stream_query：消费映射（收敛到 runner 之上后仍返回一致答案） ──────


def _neuter_accounting(monkeypatch):
    """把 stream_query 的记账副作用（audit / telemetry / 持久化）替换为 no-op。"""
    monkeypatch.setattr(cli_stream, "begin_run", lambda *a, **k: None)
    monkeypatch.setattr(cli_stream, "end_run", lambda *a, **k: None)
    monkeypatch.setattr(cli_stream, "flush_pending", lambda *a, **k: 0)
    monkeypatch.setattr(cli_stream, "_open_root_span", lambda *a, **k: None)
    monkeypatch.setattr(cli_stream, "_close_root_span", lambda *a, **k: None)

    class _FakePM:
        def upsert_meta(self, *a, **k):
            return None

    monkeypatch.setattr(cli_stream, "get_persistence", lambda: _FakePM())


def test_stream_query_returns_streamed_answer(monkeypatch):
    _neuter_accounting(monkeypatch)
    app = _FakeApp(
        legs=[[
            _node("understand", {"intent": "chat"}),
            _token("Hello "),
            _token("world"),
            _node("answer", {"final_answer": "Hello world"}),
        ]],
        state_values={"final_answer": "Hello world", "messages": []},
    )
    result = cli_stream.stream_query(app, "hi", "t1")
    assert result == "Hello world"
    assert len(app.stream_calls) == 1  # 无 interrupt：只跑一腿


def test_stream_query_prompts_confirmation_then_resumes(monkeypatch):
    _neuter_accounting(monkeypatch)
    seen = {}

    def _fake_prompt(payload):
        seen["payload"] = payload
        return {"approve": True}

    monkeypatch.setattr(cli_stream, "_prompt_confirmation", _fake_prompt)

    payload = {"type": "confirm_shell", "command": "ls"}
    app = _FakeApp(
        legs=[
            [_node("shell_plan", {"pending_shell": {"1": {}}}), _interrupt(payload)],
            [_node("answer", {"final_answer": "done"})],
        ],
        state_values={"final_answer": "done", "messages": []},
    )
    result = cli_stream.stream_query(app, "run ls", "t1")
    # 确认菜单收到了原始 interrupt 载荷；resume 后第二腿给出最终答案。
    assert seen["payload"] == payload
    assert result == "done"
    assert len(app.stream_calls) == 2  # 一腿到 interrupt，resume 后再一腿
