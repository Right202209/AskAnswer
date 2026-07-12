"""工具注册表：intent tag 过滤、confirmation_class 元数据、子图工具已 seed。"""

from __future__ import annotations

from askanswer.registry import (
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


def test_descriptor_requires_confirmation_alias():
    plain = ToolDescriptor(tool=_FakeTool("x"), tags=frozenset({"chat"}), source="builtin")
    guarded = ToolDescriptor(tool=_FakeTool("y"), tags=frozenset({"chat"}),
                             source="builtin", confirmation_class="fs_write")
    assert plain.requires_confirmation is False
    assert guarded.requires_confirmation is True
