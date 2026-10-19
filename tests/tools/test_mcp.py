"""MCP 客户端管理器的核心生命周期。"""

from askanswer.tools.mcp_manager import MCPClientManager


def test_empty_manager_close_is_idempotent():
    manager = MCPClientManager()
    manager.close()
    manager.close()

    assert manager._thread.is_alive() is False
    assert manager._loop.is_closed() is True


def test_connect_alias_removed():
    assert not hasattr(MCPClientManager, "connect")


# ── MCP schema → StructuredTool ─────────────────────────────────────────

def test_zero_arg_mcp_tool_is_wrapped(monkeypatch):
    """零参 MCP 工具不再因 `_unused` 占位字段（Pydantic v2 不允许）被静默跳过。"""
    from askanswer.tools import mcp_adapter

    calls = []

    class _Mgr:
        def call_tool(self, name, args):
            calls.append((name, args))
            return "pong"

    monkeypatch.setattr(mcp_adapter, "_mcp_manager", lambda: _Mgr())
    tool = mcp_adapter.wrap_mcp_tool(
        {"name": "srv__ping", "description": "ping", "input_schema": {"type": "object"}}
    )
    assert tool is not None
    assert tool.args == {}
    assert tool.invoke({}) == "pong"
    assert calls == [("srv__ping", {})]


def test_mcp_tool_with_required_and_optional_args(monkeypatch):
    from askanswer.tools import mcp_adapter

    seen = {}

    class _Mgr:
        def call_tool(self, name, args):
            seen.update(args)
            return "ok"

    monkeypatch.setattr(mcp_adapter, "_mcp_manager", lambda: _Mgr())
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["q"],
    }
    tool = mcp_adapter.wrap_mcp_tool({"name": "srv__search", "input_schema": schema})
    assert set(tool.args) == {"q", "limit"}
    assert tool.invoke({"q": "x", "limit": 3}) == "ok"
    assert seen == {"q": "x", "limit": 3}


def test_unsupported_mcp_schema_is_skipped():
    from askanswer.tools import mcp_adapter

    schema = {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}]}}}
    assert mcp_adapter.wrap_mcp_tool({"name": "srv__bad", "input_schema": schema}) is None
