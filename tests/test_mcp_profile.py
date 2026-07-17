"""MCP profile 读写：round-trip、原子写、按名删除、损坏 JSON 容错。"""

from __future__ import annotations

import json

import pytest

from askanswer import mcp_profile


@pytest.fixture
def profile_path(tmp_path):
    return tmp_path / "mcp.json"


def test_save_and_load_round_trip(profile_path):
    record = {"name": "srv", "transport": "stdio", "command": "python", "args": ["-m", "x"]}
    mcp_profile.save_entry(record, path=profile_path)
    loaded = mcp_profile.load(profile_path)
    assert loaded == [{"name": "srv", "transport": "stdio", "command": "python",
                       "args": ["-m", "x"]}]


def test_load_missing_file_returns_empty(profile_path):
    assert mcp_profile.load(profile_path) == []


def test_save_overwrites_same_name(profile_path):
    mcp_profile.save_entry({"name": "srv", "transport": "stdio", "url": "old"}, path=profile_path)
    mcp_profile.save_entry({"name": "srv", "transport": "url", "url": "new"}, path=profile_path)
    loaded = mcp_profile.load(profile_path)
    assert len(loaded) == 1
    assert loaded[0]["transport"] == "url"
    assert loaded[0]["url"] == "new"


def test_save_keeps_distinct_names(profile_path):
    mcp_profile.save_entry({"name": "a", "transport": "stdio"}, path=profile_path)
    mcp_profile.save_entry({"name": "b", "transport": "stdio"}, path=profile_path)
    names = {r["name"] for r in mcp_profile.load(profile_path)}
    assert names == {"a", "b"}


def test_remove_entry(profile_path):
    mcp_profile.save_entry({"name": "a", "transport": "stdio"}, path=profile_path)
    mcp_profile.save_entry({"name": "b", "transport": "stdio"}, path=profile_path)
    assert mcp_profile.remove_entry("a", path=profile_path) is True
    assert {r["name"] for r in mcp_profile.load(profile_path)} == {"b"}


def test_remove_missing_returns_false(profile_path):
    mcp_profile.save_entry({"name": "a", "transport": "stdio"}, path=profile_path)
    assert mcp_profile.remove_entry("nope", path=profile_path) is False


def test_corrupt_json_is_tolerated(profile_path):
    profile_path.write_text("{not valid json", encoding="utf-8")
    assert mcp_profile.load(profile_path) == []  # 不抛异常，当空处理


def test_non_dict_servers_key_tolerated(profile_path):
    profile_path.write_text(json.dumps({"servers": "oops"}), encoding="utf-8")
    assert mcp_profile.load(profile_path) == []


def test_invalid_record_rejected(profile_path):
    with pytest.raises(ValueError):
        mcp_profile.save_entry({"name": "no-transport"}, path=profile_path)


def test_clean_record_drops_unknown_and_none(profile_path):
    mcp_profile.save_entry(
        {"name": "srv", "transport": "stdio", "junk": "x", "url": None},
        path=profile_path,
    )
    loaded = mcp_profile.load(profile_path)[0]
    assert "junk" not in loaded
    assert "url" not in loaded  # None 值被丢弃


def test_atomic_write_leaves_no_tmp_files(profile_path):
    mcp_profile.save_entry({"name": "srv", "transport": "stdio"}, path=profile_path)
    leftovers = list(profile_path.parent.glob("*.tmp"))
    assert leftovers == []


def test_load_filters_records_missing_required_fields(profile_path):
    profile_path.write_text(
        json.dumps({"servers": [{"name": "ok", "transport": "stdio"},
                                 {"name": "bad-no-transport"}]}),
        encoding="utf-8",
    )
    loaded = mcp_profile.load(profile_path)
    assert [r["name"] for r in loaded] == ["ok"]


def test_default_profile_path_honors_env(monkeypatch, tmp_path):
    target = tmp_path / "custom-mcp.json"
    monkeypatch.setenv("ASKANSWER_MCP_PROFILE", str(target))
    assert mcp_profile.default_profile_path() == target


# ── 安全 / 白名单补充 ────────────────────────────────────────────────────

def test_headers_field_preserved(profile_path):
    """headers 属于白名单字段，脏字段被丢弃的同时它必须原样保留。"""
    mcp_profile.save_entry(
        {"name": "srv", "transport": "stdio", "command": "python",
         "evil": "rm -rf", "headers": {"a": "b"}},
        path=profile_path,
    )
    loaded = mcp_profile.load(profile_path)[0]
    assert "evil" not in loaded
    assert loaded["headers"] == {"a": "b"}


def test_profile_file_mode_is_owner_only(profile_path):
    """profile 可含 Authorization 令牌，落盘必须 0o600（owner-only）。

    当前由 ``mkstemp`` 默认 0o600 + ``os.replace`` 保留 inode 权限保证；
    本测试锁定该不变量，防止将来改成直接 ``open()`` 写入而降权。
    """
    import stat

    mcp_profile.save_entry(
        {"name": "srv", "transport": "stdio",
         "headers": {"Authorization": "Bearer t"}},
        path=profile_path,
    )
    assert stat.S_IMODE(profile_path.stat().st_mode) == 0o600
