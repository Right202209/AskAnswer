"""确认协议：按 class 分发、fs_write 危险路径全拒、plan→gate→interrupt→apply 四步。"""

from __future__ import annotations

import pytest
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from askanswer.config.schema import ContextSchema
from askanswer.core import confirmations as C

# ── handler 分发 ─────────────────────────────────────────────────────────

def test_supported_classes():
    assert C.supported_confirmation_classes() == frozenset(
        {"shell", "fs_write", "external_api_paid"}
    )


@pytest.mark.parametrize(
    "clazz,expected",
    [
        ("shell", C.ShellConfirmation),
        ("fs_write", C.FsWriteConfirmation),
        ("external_api_paid", C.PaidApiConfirmation),
    ],
)
def test_get_handler_dispatches_by_class(clazz, expected):
    assert isinstance(C.get_confirmation_handler(clazz), expected)


@pytest.mark.parametrize("clazz", [None, "", "none", "unknown"])
def test_get_handler_none_for_unmapped(clazz):
    assert C.get_confirmation_handler(clazz) is None


# ── fs_write 危险路径全拒 ────────────────────────────────────────────────

@pytest.mark.parametrize(
    "path",
    [
        "~/.ssh/authorized_keys",
        "/tmp/secret.pem",
        "/tmp/id_rsa",
        "/tmp/service.key",
        "~/.netrc",
    ],
)
def test_fs_write_plan_rejects_sensitive_paths(path):
    handler = C.FsWriteConfirmation()
    payload = handler.plan({"name": "write_file", "args": {"path": path, "content": "x"}})
    assert "error" in payload  # 敏感路径在 plan 阶段即被拦截


def test_fs_write_plan_rejects_oversize(monkeypatch, tmp_path):
    # 把上限压到 10 字节，写 11 字节应被拒
    monkeypatch.setattr("askanswer.tools.filesystem._WRITE_FILE_MAX_BYTES", 10)
    handler = C.FsWriteConfirmation()
    payload = handler.plan(
        {"name": "write_file", "args": {"path": str(tmp_path / "f.txt"), "content": "x" * 11}}
    )
    assert "error" in payload


def test_fs_write_apply_denied_does_not_write(tmp_path):
    target = tmp_path / "out.txt"
    handler = C.FsWriteConfirmation()
    payload = handler.plan(
        {"name": "write_file", "args": {"path": str(target), "content": "hello"}}
    )
    outcome = handler.apply(payload, decision=False,
                            tool_call={"args": {"path": str(target), "content": "hello"}})
    assert outcome.approved is False
    assert not target.exists()  # 拒绝 → 不落盘（不变量 7）


def test_fs_write_apply_approved_writes(tmp_path):
    target = tmp_path / "out.txt"
    handler = C.FsWriteConfirmation()
    call = {"name": "write_file", "args": {"path": str(target), "content": "hello"}}
    payload = handler.plan(call)
    outcome = handler.apply(payload, decision=True, tool_call=call)
    assert outcome.approved is True
    assert target.read_text(encoding="utf-8") == "hello"


# ── shell gate 拦截危险命令 ──────────────────────────────────────────────

def test_shell_gate_blocks_dangerous():
    handler = C.ShellConfirmation()
    blocked = handler.gate({"command": "rm -rf /", "explanation": ""})
    assert blocked is not None
    assert "rm" in blocked


def test_shell_gate_allows_safe():
    handler = C.ShellConfirmation()
    assert handler.gate({"command": "ls -la", "explanation": ""}) is None


def test_shell_apply_rechecks_edited_command():
    """用户批准时把命令改成危险的，apply 必须重新拦截（双重检查）。"""
    handler = C.ShellConfirmation()
    outcome = handler.apply(
        {"command": "ls"}, decision={"approve": True, "command": "rm -rf /"},
        tool_call={},
    )
    assert outcome.approved is False
    assert outcome.error is not None


def test_shell_interrupt_payload_type():
    handler = C.ShellConfirmation()
    payload = handler.interrupt_payload({"command": "ls", "explanation": "list", "instruction": "看看"})
    assert payload["type"] == "confirm_shell"
    assert payload["command"] == "ls"


# ── 决策解析 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "decision,expected",
    [
        (True, True),
        (False, False),
        ("yes", True),
        ("y", True),
        ("1", True),
        ("no", False),
        (None, False),
        ({"approve": True}, True),
        ({"value": "yes"}, True),
    ],
)
def test_parse_approval(decision, expected):
    assert C.parse_approval(decision) is expected


def test_parse_decision_preserves_edited_command():
    approved, command = C.parse_decision({"approve": True, "command": "echo hi"}, "fallback")
    assert approved is True
    assert command == "echo hi"


# ── external_api_paid：批准后经 ToolNode 执行，ToolRuntime[ContextSchema] 照常注入 ──

_PAID_CALLS: list = []


@tool
def paid_lookup(q: str, runtime: ToolRuntime[ContextSchema]) -> str:
    """付费查询（测试用）。"""
    tenant = getattr(runtime.context, "tenant_id", None)
    _PAID_CALLS.append((q, tenant))
    if q == "boom":
        raise RuntimeError("quota exceeded")
    return f"{q}@{tenant}"


@pytest.fixture
def paid_tool_registered():
    from askanswer.tools.registry import ToolDescriptor, get_registry

    _PAID_CALLS.clear()
    registry = get_registry()
    registry.register(
        ToolDescriptor(tool=paid_lookup, tags=frozenset({"chat"}), source="test",
                       confirmation_class="external_api_paid")
    )
    yield _PAID_CALLS
    registry.unregister_source_prefix("test")


def _apply_inside_graph(call, decision, *, tenant):
    """在真实 LangGraph 节点里调用 apply（与 tools 节点相同的运行时上下文）。"""
    from langgraph.graph import END, START, MessagesState, StateGraph

    handler = C.PaidApiConfirmation()
    payload = handler.plan(call)
    holder = {}

    def node(state):
        holder["outcome"] = handler.apply(payload, decision, call)
        return {}

    graph = StateGraph(MessagesState, context_schema=ContextSchema)
    graph.add_node("n", node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    graph.compile().invoke({"messages": []}, context=ContextSchema(tenant_id=tenant))
    return holder["outcome"]


def test_paid_apply_injects_runtime_context(paid_tool_registered):
    call = {"name": "paid_lookup", "args": {"q": "x"}, "id": "c1"}
    outcome = _apply_inside_graph(call, True, tenant="acme")
    assert outcome.approved is True and outcome.error is None
    assert outcome.content == "x@acme"
    assert paid_tool_registered == [("x", "acme")]


def test_paid_apply_tool_error_is_reported(paid_tool_registered):
    call = {"name": "paid_lookup", "args": {"q": "boom"}, "id": "c2"}
    outcome = _apply_inside_graph(call, True, tenant="acme")
    assert outcome.approved is True
    assert outcome.error and "quota exceeded" in outcome.content


def test_paid_apply_rejected_does_not_call(paid_tool_registered):
    call = {"name": "paid_lookup", "args": {"q": "x"}, "id": "c3"}
    outcome = _apply_inside_graph(call, False, tenant="acme")
    assert outcome.approved is False
    assert paid_tool_registered == []


def test_paid_api_hitl_end_to_end_resume(fake_llm, paid_tool_registered):
    """完整 HITL：answer 规划付费调用 → interrupt → resume 批准 → 工具拿到租户上下文。"""
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from askanswer import create_search_assistant, resume_input, run_leg
    from askanswer.core.runner import EVENT_FINAL, EVENT_INTERRUPT, query_input

    fake_llm.responses = [
        AIMessage(content="", tool_calls=[
            {"name": "paid_lookup", "args": {"q": "fx"}, "id": "call-1", "type": "tool_call"}
        ]),
        "汇率已查询",
    ]
    app = create_search_assistant(InMemorySaver())
    ctx = ContextSchema(tenant_id="acme")

    first = list(run_leg(app, query_input("你好，查一下"), thread_id="paid-1", context=ctx))
    assert first[-1].kind == EVENT_INTERRUPT
    assert first[-1].data["type"] == "confirm_external_api_paid"
    assert first[-1].data["tool"] == "paid_lookup"
    assert paid_tool_registered == []  # 批准前不执行

    second = list(run_leg(app, resume_input({"approve": True}), thread_id="paid-1", context=ctx))
    assert second[-1].kind == EVENT_FINAL
    assert second[-1].text == "汇率已查询"
    assert paid_tool_registered == [("fx", "acme")]
    state = app.get_state({"configurable": {"thread_id": "paid-1"}}).values
    tool_messages = [m for m in state["messages"] if getattr(m, "type", "") == "tool"]
    assert tool_messages[-1].content == "fx@acme"
    assert state["pending_confirmations"] == {}
