"""UI 无关的 runner 事件流契约。"""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk

from askanswer.core import runner


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


# ── 持久化后端解析：InMemorySaver → NullPersistence，不创建 SQLite ─────────────

def test_resolve_persistence_defaults_to_null_for_foreign_checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver

    from askanswer.storage import NULL_PERSISTENCE

    class _App:
        checkpointer = InMemorySaver()

    assert runner.resolve_persistence(_App()) is NULL_PERSISTENCE
    assert runner.resolve_persistence(object()) is NULL_PERSISTENCE


def test_resolve_persistence_finds_manager_by_saver(pm):
    class _App:
        checkpointer = pm.checkpointer

    assert runner.resolve_persistence(_App()) is pm


def test_resolve_persistence_explicit_wins(pm):
    sentinel = object()

    class _App:
        checkpointer = pm.checkpointer

    assert runner.resolve_persistence(_App(), sentinel) is sentinel


def test_run_leg_writes_to_explicit_backend():
    class _Recorder:
        def __init__(self):
            self.meta = []
            self.events = []

        def upsert_meta(self, thread_id, **fields):
            self.meta.append((thread_id, fields))

        def log_audit_event(self, thread_id, **fields):
            self.events.append(fields)

        def log_audit_events(self, events):
            self.events.extend(events)
            return len(events)

    backend = _Recorder()
    app = _FakeApp([_node("answer", {"final_answer": "done"})], {"intent": "chat"})
    events = list(
        runner.run_leg(
            app,
            runner.query_input("hi"),
            thread_id="t-explicit",
            context=runner.runtime_context_from_env(),
            preview="hi",
            persistence=backend,
        )
    )
    assert events[-1].kind == runner.EVENT_FINAL
    assert backend.meta and backend.meta[0][0] == "t-explicit"
    assert backend.meta[0][1]["preview"] == "hi"


def test_run_leg_with_inmemory_saver_creates_no_sqlite(fake_llm, monkeypatch, tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver

    import askanswer.storage.manager as manager
    from askanswer import ContextSchema, create_search_assistant

    home = tmp_path / "home"
    db_path = tmp_path / "state.db"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ASKANSWER_DB_PATH", str(db_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    fake_llm.responses = ["你好，我是 AskAnswer。"]

    app = create_search_assistant(InMemorySaver())
    events = list(
        runner.run_leg(
            app, runner.query_input("你好"), thread_id="mem-1", context=ContextSchema()
        )
    )

    assert events[-1].kind == runner.EVENT_FINAL
    assert events[-1].text == "你好，我是 AskAnswer。"
    assert not db_path.exists()
    assert not (home / ".askanswer").exists()
    assert manager._singleton is None


def test_run_leg_with_manager_checkpointer_persists_meta_and_audit(fake_llm, pm):
    from askanswer import ContextSchema, create_search_assistant

    fake_llm.responses = ["hello there"]
    app = create_search_assistant(pm.checkpointer)
    list(
        runner.run_leg(
            app, runner.query_input("hello"), thread_id="db-1",
            context=ContextSchema(tenant_id="acme"), preview="hello",
        )
    )
    meta = pm.get_meta("db-1")
    assert meta is not None
    assert meta.message_count == 1
    assert meta.tenant_id == "acme"
    llm_events = pm.list_audit_events(thread_id="db-1", kind="llm_call")
    assert llm_events and all(e.tenant_id == "acme" for e in llm_events)
