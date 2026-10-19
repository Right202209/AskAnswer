"""工具注册表：intent tag 过滤、confirmation_class 元数据、子图工具已 seed。"""

from __future__ import annotations

from askanswer.tools.registry import (
    ALL_INTENT_TAGS,
    ToolDescriptor,
    ToolRegistry,
    get_registry,
)

# ── seed 内容 ────────────────────────────────────────────────────────────

def test_registry_seeds_subgraph_tools():
    names = get_registry().names()
    for expected in ("sql_query", "helix_spec_loop", "research_brief_loop",
                     "decision_memo_loop"):
        assert expected in names, f"缺少种子工具 {expected}"


def test_registry_seeds_builtin_tools():
    names = get_registry().names()
    for expected in ("read_file", "write_file", "tavily_search", "calculate"):
        assert expected in names


def test_all_intent_tags_count():
    # 8 个 intent tag 与 handler 数量对齐
    assert len(ALL_INTENT_TAGS) == 8


# ── tag 过滤 ─────────────────────────────────────────────────────────────

def test_list_by_tag_filters():
    reg = get_registry()
    sql_tools = {t.name for t in reg.list(tags={"sql"})}
    # sql_query 暴露给 sql tag；write_file 不应出现（仅 chat/file_read/fs_write）
    assert "sql_query" in sql_tools
    assert "write_file" not in sql_tools


def test_write_file_scoped_to_chat_and_file_read():
    reg = get_registry()
    file_tools = {t.name for t in reg.list(tags={"file_read"})}
    assert "write_file" in file_tools
    sql_tools = {t.name for t in reg.list(tags={"sql"})}
    assert "write_file" not in sql_tools


# ── confirmation_class 元数据 ────────────────────────────────────────────

def test_confirmation_classes_metadata():
    reg = get_registry()
    classes = reg.confirmation_classes()
    assert classes.get("write_file") == "fs_write"
    assert classes.get("gen_shell_commands_run") == "shell"
    # 普通只读工具不需要确认
    assert "read_file" not in classes
    assert "calculate" not in classes


def test_confirmation_names_subset_of_classes():
    reg = get_registry()
    assert set(reg.confirmation_names()) == set(reg.confirmation_classes())


# ── ToolRegistry 基本操作（隔离实例，不碰全局单例） ──────────────────────

class _FakeTool:
    def __init__(self, name):
        self.name = name


def test_register_and_get():
    reg = ToolRegistry()
    desc = ToolDescriptor(tool=_FakeTool("x"), tags=frozenset({"chat"}), source="builtin")
    reg.register(desc)
    assert reg.get("x") is desc


def test_register_overwrites_same_name():
    reg = ToolRegistry()
    reg.register(ToolDescriptor(tool=_FakeTool("x"), tags=frozenset({"chat"}), source="a"))
    reg.register(ToolDescriptor(tool=_FakeTool("x"), tags=frozenset({"sql"}), source="b"))
    assert reg.get("x").source == "b"


def test_unregister_source_prefix():
    reg = ToolRegistry()
    reg.register(ToolDescriptor(tool=_FakeTool("m1"), tags=frozenset({"chat"}), source="mcp:s1"))
    reg.register(ToolDescriptor(tool=_FakeTool("b1"), tags=frozenset({"chat"}), source="builtin"))
    reg.unregister_source_prefix("mcp:")
    assert reg.get("m1") is None
    assert reg.get("b1") is not None


def test_descriptor_confirmation_class_default_and_routing():
    reg = ToolRegistry()
    reg.register(ToolDescriptor(tool=_FakeTool("x"), tags=frozenset({"chat"}), source="builtin"))
    reg.register(ToolDescriptor(tool=_FakeTool("y"), tags=frozenset({"chat"}),
                                source="builtin", confirmation_class="fs_write"))
    assert reg.get("x").confirmation_class == "none"
    assert reg.confirmation_classes() == {"y": "fs_write"}


def test_legacy_aliases_removed():
    import askanswer.tools.registry as registry

    for name in ("BUNDLE_CHAT", "BUNDLE_SEARCH", "BUNDLE_FILE", "BUNDLE_SQL", "ALL_BUNDLES"):
        assert not hasattr(registry, name)
    assert not hasattr(ToolDescriptor, "bundles")
    assert not hasattr(ToolDescriptor, "requires_confirmation")


# ── 外部客户端惰性构造：导入期不读凭据 ────────────────────────────────────

def test_tavily_client_is_lazy_and_reads_env_at_first_use(monkeypatch):
    from askanswer.tools import builtin

    builtin.reset_tavily_client()
    assert builtin._tavily_client is None  # 导入后尚未构造
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-late")
    client = builtin.get_tavily_client()
    assert client is builtin.get_tavily_client()  # 进程内复用
    assert getattr(client, "api_key", "tvly-late") == "tvly-late"
    builtin.reset_tavily_client()


def test_openweather_key_read_at_call_time(monkeypatch):
    from askanswer.tools import builtin

    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    assert builtin.check_weather.invoke({"city": "Beijing"}) == "未配置 OPENWEATHER_API_KEY"
    monkeypatch.setenv("OPENWEATHER_API_KEY", "ow-late")
    assert builtin.openweather_api_key() == "ow-late"


def test_size_limits_read_env_at_call_time(monkeypatch):
    from askanswer.tools import filesystem, shell

    monkeypatch.setenv("ASKANSWER_WRITE_FILE_MAX_BYTES", "123")
    monkeypatch.setenv("ASKANSWER_READ_FILE_MAX_BYTES", "456")
    monkeypatch.setenv("ASKANSWER_SHELL_OUTPUT_MAX_BYTES", "789")
    assert filesystem.write_file_max_bytes() == 123
    assert filesystem.read_file_max_bytes() == 456
    assert shell.shell_output_max_bytes() == 789


# ── ASKANSWER_MCP_ALL_INTENTS 动态读取 ──────────────────────────────────────

def _registry_with_fake_mcp(monkeypatch):
    from askanswer.tools import mcp_adapter, registry

    class _Mgr:
        def list_tools(self):
            return [{"name": "srv__echo", "server": "srv", "description": "echo",
                     "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}]

        def list_servers(self):
            return [{"name": "srv", "status": "connected"}]

        def call_tool(self, name, args):
            return "ok"

    monkeypatch.setattr(registry, "_mcp_manager", lambda: _Mgr())
    monkeypatch.setattr(mcp_adapter, "_mcp_manager", lambda: _Mgr())
    return ToolRegistry()


def test_mcp_tags_default_chat_only(monkeypatch):
    monkeypatch.delenv("ASKANSWER_MCP_ALL_INTENTS", raising=False)
    reg = _registry_with_fake_mcp(monkeypatch)
    reg.refresh_mcp()
    assert "srv__echo" in reg.names(tags={"chat"})
    assert "srv__echo" not in reg.names(tags={"sql"})


def test_mcp_all_intents_read_at_refresh_time(monkeypatch):
    monkeypatch.delenv("ASKANSWER_MCP_ALL_INTENTS", raising=False)
    reg = _registry_with_fake_mcp(monkeypatch)
    reg.refresh_mcp()
    assert "srv__echo" not in reg.names(tags={"sql"})
    # 进程运行中打开开关，再 refresh 即生效（以前是导入期常量，不会变）
    monkeypatch.setenv("ASKANSWER_MCP_ALL_INTENTS", "1")
    reg.refresh_mcp()
    assert "srv__echo" in reg.names(tags={"sql"})
