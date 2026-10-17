"""MCP 客户端管理器的核心生命周期。"""

from askanswer.mcp import MCPClientManager


def test_empty_manager_close_is_idempotent():
    manager = MCPClientManager()
    manager.close()
    manager.close()

    assert manager._thread.is_alive() is False
    assert manager._loop.is_closed() is True
