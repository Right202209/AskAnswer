"""工具注册表：种子工具就位、tag 过滤、confirmation_class 元数据。"""

from __future__ import annotations

from askanswer.registry import get_registry


def test_seed_tools_present():
    names = get_registry().names()
    for expected in (
        "research_brief_loop",
        "decision_memo_loop",
        "helix_spec_loop",
    ):
        assert expected in names, f"缺少种子工具 {expected}"


def test_tag_filter_returns_subset():
    registry = get_registry()
    everything = registry.list()
    research = registry.list(tags={"research"})
    assert research, "research tag 应至少命中 research_brief_loop"
    assert len(research) <= len(everything)
    assert all(hasattr(t, "name") for t in research)


def test_tag_filter_miss_returns_empty():
    assert get_registry().list(tags={"no-such-tag-xyz"}) == []


def test_confirmation_metadata_shape():
    registry = get_registry()
    classes = registry.confirmation_classes()
    # 每个需要确认的工具都有非 none 的类，且类名在已知集合内
    known = {"shell", "fs_write", "external_api_paid"}
    assert set(classes.values()) <= known
    assert set(classes) == registry.confirmation_names()


def test_write_file_is_fs_write_class():
    descriptor = get_registry().get("write_file")
    if descriptor is not None:  # write_file 为内置工具
        assert descriptor.confirmation_class == "fs_write"
        assert descriptor.requires_confirmation is True
