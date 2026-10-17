"""UI 无关的 runner 事件流契约。"""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk

from askanswer import runner


class _FakeState:
    def __init__(self, values: dict, tasks=()):
        self.values = values
        self.tasks = tasks


class _FakeApp:
    def __init__(self, chunks: list[tuple[str, object]], state_values: dict):
        self._chunks = chunks
        self._state_values = state_values

    def stream(self, graph_input, *, config, context, stream_mode):
        yield from self._chunks

    def get_state(self, config):
        return _FakeState(self._state_values)


def _token(text: str) -> tuple[str, tuple]:
    return "messages", (AIMessageChunk(content=text), {})


def _tool_plan() -> tuple[str, tuple]:
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "search", "args": "", "id": "1", "index": 0}],
    )
    return "messages", (chunk, {})


def _node(name: str, update: dict) -> tuple[str, dict]:
    return "updates", {name: update}


def test_stream_leg_emits_tokens_tools_nodes_and_final():
    app = _FakeApp(
        [
            _node("understand", {"intent": "chat"}),
            _tool_plan(),
            _token("Hello "),
            _token("world"),
            _node("answer", {"final_answer": "Hello world"}),
        ],
        {"final_answer": "Hello world", "messages": []},
    )

    events = list(
        runner.stream_leg(
            app,
            runner.query_input("hi"),
            config=runner.thread_config("t1"),
            context=runner.runtime_context_from_env(),
        )
    )

    assert [event.kind for event in events] == [
        runner.EVENT_NODE,
        runner.EVENT_TOOL,
        runner.EVENT_TOKEN,
        runner.EVENT_TOKEN,
        runner.EVENT_NODE,
        runner.EVENT_FINAL,
    ]
    assert events[-1].text == "Hello world"


def test_stream_leg_interrupt_has_no_final_event():
    payload = {"type": "confirm_shell", "command": "ls"}
    app = _FakeApp(
        [("updates", {"__interrupt__": payload})],
        {},
    )

    events = list(
        runner.stream_leg(
            app,
            runner.query_input("run ls"),
            config=runner.thread_config("t1"),
            context=runner.runtime_context_from_env(),
        )
    )

    assert events[-1].kind == runner.EVENT_INTERRUPT
    assert events[-1].data == payload
    assert all(event.kind != runner.EVENT_FINAL for event in events)
