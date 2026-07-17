"""mcp_profile：round-trip、原子写、按名删除、损坏文件容错、字段白名单。"""

from __future__ import annotations

import json

import pytest

from askanswer import mcp_profile


def _entry(name="srv", **extra):
    return {"name": name, "transport": "stdio", "command": "python", **extra}


def test_round_trip(tmp_path):
    path = tmp_path / "mcp.json"
    mcp_profile.save_entry(_entry(args=["-m", "askanswer.helix_mcp"]), path=path)
    loaded = mcp_profile.load(path)
    assert len(loaded) == 1
    assert loaded[0]["name"] == "srv"
    assert loaded[0]["args"] == ["-m", "askanswer.helix_mcp"]


def test_save_overwrites_same_name(tmp_path):
    path = tmp_path / "mcp.json"
    mcp_profile.save_entry(_entry(command="old"), path=path)
    mcp_profile.save_entry(_entry(command="new"), path=path)
    loaded = mcp_profile.load(path)
    assert len(loaded) == 1
    assert loaded[0]["command"] == "new"


def test_remove_entry(tmp_path):
    path = tmp_path / "mcp.json"
    mcp_profile.save_entry(_entry("a"), path=path)
    mcp_profile.save_entry(_entry("b"), path=path)
    assert mcp_profile.remove_entry("a", path=path) is True
    assert [r["name"] for r in mcp_profile.load(path)] == ["b"]
    assert mcp_profile.remove_entry("missing", path=path) is False


def test_corrupt_file_tolerated(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("{broken json", encoding="utf-8")
    assert mcp_profile.load(path) == []


def test_missing_file_returns_empty(tmp_path):
    assert mcp_profile.load(tmp_path / "absent.json") == []


def test_atomic_write_no_tmp_leftovers(tmp_path):
    path = tmp_path / "mcp.json"
    mcp_profile.save_entry(_entry(), path=path)
    assert not list(tmp_path.glob("*.tmp"))
    # 文件内容是合法 JSON 且带 servers 键
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data["servers"], list)


def test_unknown_fields_dropped(tmp_path):
    path = tmp_path / "mcp.json"
    mcp_profile.save_entry(_entry(evil="rm -rf", headers={"a": "b"}), path=path)
    loaded = mcp_profile.load(path)[0]
    assert "evil" not in loaded
    assert loaded["headers"] == {"a": "b"}


def test_invalid_record_rejected(tmp_path):
    with pytest.raises(ValueError):
        mcp_profile.save_entry({"name": "x"}, path=tmp_path / "mcp.json")


def test_profile_file_mode_is_owner_only(tmp_path):
    import stat

    path = tmp_path / "mcp.json"
    mcp_profile.save_entry(_entry(headers={"Authorization": "Bearer t"}), path=path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
