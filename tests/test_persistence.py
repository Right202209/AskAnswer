"""持久化层：v4 建库、v2→v4 无损升级、tenant 过滤、upsert COALESCE、checkpoint label。"""

from __future__ import annotations

import sqlite3
import time

import pytest

from askanswer.persistence import PersistenceManager, _SCHEMA_VERSION


def _schema_version(db_path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM askanswer_schema WHERE key='version'"
        ).fetchone()
        return int(row[0]) if row else -1
    finally:
        conn.close()


def _table_columns(db_path, table) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def _tables(db_path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


def test_fresh_db_is_v4(pm):
    assert _SCHEMA_VERSION == 4
    assert _schema_version(pm.db_path) == 4
    assert {"thread_meta", "audit_event", "checkpoint_label"} <= _tables(pm.db_path)
    assert "tenant_id" in _table_columns(pm.db_path, "thread_meta")
    assert "tenant_id" in _table_columns(pm.db_path, "audit_event")


def _build_v2_db(path):
    """手工构造一个 v2 库（无 tenant_id / checkpoint_label），写入真实数据。"""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE askanswer_schema(key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO askanswer_schema VALUES('version','2');
        CREATE TABLE thread_meta(thread_id TEXT PRIMARY KEY, title TEXT, tags TEXT,
          created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
          message_count INTEGER NOT NULL DEFAULT 0, last_intent TEXT,
          model_label TEXT, preview TEXT);
        CREATE TABLE audit_event(id INTEGER PRIMARY KEY AUTOINCREMENT,
          thread_id TEXT NOT NULL, ts INTEGER NOT NULL, kind TEXT NOT NULL,
          tool_name TEXT, args_summary TEXT, result_size INTEGER, model_label TEXT,
          input_tokens INTEGER, output_tokens INTEGER, duration_ms INTEGER,
          intent TEXT, error TEXT);
        """
    )
    now = int(time.time())
    conn.execute(
        "INSERT INTO thread_meta VALUES(?,?,?,?,?,?,?,?,?)",
        ("legacy-1", "Legacy Title", "[]", now, now, 7, "search", "gpt", "preview"),
    )
    conn.execute(
        "INSERT INTO audit_event(thread_id,ts,kind,tool_name) VALUES(?,?,?,?)",
        ("legacy-1", now, "tool_call", "tavily_search"),
    )
    conn.commit()
    conn.close()


def test_v2_upgrades_to_v4_losslessly(tmp_path):
    db = tmp_path / "old.db"
    _build_v2_db(db)
    manager = PersistenceManager(db)
    try:
        meta = manager.get_meta("legacy-1")
        assert meta is not None
        assert meta.title == "Legacy Title"
        assert meta.message_count == 7
        assert meta.tenant_id is None  # 旧行迁移后归属为空
        events = manager.list_audit_events(thread_id="legacy-1")
        assert len(events) == 1
        assert events[0].tool_name == "tavily_search"
    finally:
        manager.close()
    assert _schema_version(db) == 4
    assert "checkpoint_label" in _tables(db)
    assert "tenant_id" in _table_columns(db, "thread_meta")


def test_legacy_null_tenant_row_hidden_from_named_tenant(tmp_path):
    db = tmp_path / "old.db"
    _build_v2_db(db)
    manager = PersistenceManager(db)
    try:
        assert [t.thread_id for t in manager.list_threads()] == ["legacy-1"]
        assert manager.list_threads(tenant_id="alice") == []
    finally:
        manager.close()


def test_tenant_filter_list_and_get(pm):
    pm.upsert_meta("th-a", title="a", preview="pa", tenant_id="alice")
    pm.upsert_meta("th-b", title="b", preview="pb", tenant_id="bob")
    assert [t.thread_id for t in pm.list_threads(tenant_id="alice")] == ["th-a"]
    assert {t.thread_id for t in pm.list_threads()} == {"th-a", "th-b"}
    assert pm.get_meta("th-a", tenant_id="alice") is not None
    assert pm.get_meta("th-a", tenant_id="bob") is None
    assert pm.get_meta("th-a") is not None  # None = 不限租户


def test_find_by_prefix_respects_tenant(pm):
    pm.upsert_meta("abcd-alice-thread", title="a", tenant_id="alice")
    pm.upsert_meta("abcd-bob-thread", title="b", tenant_id="bob")
    hits = pm.find_by_prefix("abcd", tenant_id="alice")
    assert [t.thread_id for t in hits] == ["abcd-alice-thread"]
    assert len(pm.find_by_prefix("abcd")) == 2
    assert pm.find_by_prefix("ab") == []  # <4 字符不生效


def test_cross_tenant_delete_is_blocked(pm):
    pm.upsert_meta("th-a", title="a", tenant_id="alice")
    assert pm.delete_thread("th-a", tenant_id="bob") is False
    assert pm.get_meta("th-a") is not None  # 未被误删
    assert pm.delete_thread("th-a", tenant_id="alice") is True
    assert pm.get_meta("th-a") is None


def test_audit_events_tenant_filter(pm):
    pm.log_audit_event("th-a", kind="tool_call", tool_name="t1", tenant_id="alice")
    pm.log_audit_event("th-b", kind="tool_call", tool_name="t2", tenant_id="bob")
    alice = pm.list_audit_events(tenant_id="alice")
    assert [e.tool_name for e in alice] == ["t1"]
    assert len(pm.list_audit_events()) == 2


def test_upsert_coalesce_preserves_values(pm):
    pm.upsert_meta("th", title="First Title", preview="p1", message_count=3)
    # 后续 title=None 不得覆盖既有标题；preview 非 None 才更新
    pm.upsert_meta("th", intent="search", preview="p2")
    meta = pm.get_meta("th")
    assert meta.title == "First Title"  # 保留
    assert meta.preview == "p2"  # 更新
    assert meta.message_count == 3  # message_count=None 不动
    assert meta.last_intent == "search"


def test_upsert_title_derived_from_preview_on_insert(pm):
    pm.upsert_meta("th", preview="hello world preview")
    meta = pm.get_meta("th")
    assert meta.title  # 首次插入无 title 时从 preview 派生


def test_upsert_does_not_steal_tenant(pm):
    pm.upsert_meta("th", title="t", tenant_id="alice")
    # 另一个租户后写同一 thread，不能把归属抢走（COALESCE 保留既有）
    pm.upsert_meta("th", preview="x", tenant_id="bob")
    assert pm.get_meta("th").tenant_id == "alice"


def test_checkpoint_label_set_resolve_move(pm):
    assert pm.set_checkpoint_label("th", "ckpt-1", "before-edit") is True
    assert pm.resolve_checkpoint_label("th", "before-edit") == "ckpt-1"
    # 同名 label 再打 => 移动指向新 checkpoint
    assert pm.set_checkpoint_label("th", "ckpt-2", "before-edit") is True
    assert pm.resolve_checkpoint_label("th", "before-edit") == "ckpt-2"
    assert len(pm.list_checkpoint_labels("th")) == 1


@pytest.mark.parametrize("bad", [("", "c", "l"), ("th", "", "l"), ("th", "c", "")])
def test_checkpoint_label_rejects_empty(pm, bad):
    assert pm.set_checkpoint_label(*bad) is False


def test_resolve_missing_label_returns_none(pm):
    assert pm.resolve_checkpoint_label("th", "nope") is None
