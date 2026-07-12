"""持久化层：v4 建库、v2→v4 迁移无损、多租户过滤、UPSERT COALESCE、checkpoint label。

这些是两次 schema 迁移与多租户过滤的回归保护 —— 全部在临时 SQLite 上跑，不依赖 API key。
"""

from __future__ import annotations

import sqlite3

import pytest

from askanswer.persistence import PersistenceManager


def _schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM askanswer_schema WHERE key = 'version'"
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


# ── 空库建到 v4 ─────────────────────────────────────────────────────────

def test_fresh_db_is_v4(pm):
    assert _schema_version(pm._conn) == 4


def test_fresh_db_has_tenant_columns_and_label_table(pm):
    assert "tenant_id" in _columns(pm._conn, "thread_meta")
    assert "tenant_id" in _columns(pm._conn, "audit_event")
    assert _table_exists(pm._conn, "checkpoint_label")


# ── v2 旧库 → v4 迁移无损 ────────────────────────────────────────────────

def _build_v2_db(path) -> None:
    """手工造一个 schema v2 的旧库（无 tenant_id / 无 checkpoint_label），塞入一行数据。"""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE askanswer_schema (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO askanswer_schema(key, value) VALUES ('version', '2');
        CREATE TABLE thread_meta (
            thread_id TEXT PRIMARY KEY, title TEXT, tags TEXT,
            created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            last_intent TEXT, model_label TEXT, preview TEXT
        );
        CREATE TABLE audit_event (
            id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,
            ts INTEGER NOT NULL, kind TEXT NOT NULL, tool_name TEXT,
            args_summary TEXT, result_size INTEGER, model_label TEXT,
            input_tokens INTEGER, output_tokens INTEGER, duration_ms INTEGER,
            intent TEXT, error TEXT
        );
        INSERT INTO thread_meta(thread_id, title, tags, created_at, updated_at, message_count)
            VALUES ('old-thread', '旧标题', '[]', 100, 200, 3);
        INSERT INTO audit_event(thread_id, ts, kind, tool_name)
            VALUES ('old-thread', 150, 'tool_call', 'read_file');
        """
    )
    conn.commit()
    conn.close()


def test_v2_upgrades_to_v4(tmp_path):
    db = tmp_path / "legacy.db"
    _build_v2_db(db)
    pm = PersistenceManager(db)
    try:
        assert _schema_version(pm._conn) == 4
        assert "tenant_id" in _columns(pm._conn, "thread_meta")
        assert _table_exists(pm._conn, "checkpoint_label")
    finally:
        pm.close()


def test_v2_data_survives_migration(tmp_path):
    db = tmp_path / "legacy.db"
    _build_v2_db(db)
    pm = PersistenceManager(db)
    try:
        meta = pm.get_meta("old-thread")
        assert meta is not None
        assert meta.title == "旧标题"
        assert meta.message_count == 3
        assert meta.tenant_id is None  # 迁移后旧行 tenant 为 NULL，不退化行为
        events = pm.list_audit_events(thread_id="old-thread")
        assert len(events) == 1
        assert events[0].tool_name == "read_file"
    finally:
        pm.close()


# ── 多租户过滤 ───────────────────────────────────────────────────────────

def test_list_threads_filters_by_tenant(pm):
    pm.upsert_meta("t-alice", title="A", tenant_id="alice")
    pm.upsert_meta("t-bob", title="B", tenant_id="bob")
    alice = {m.thread_id for m in pm.list_threads(tenant_id="alice")}
    assert alice == {"t-alice"}
    bob = {m.thread_id for m in pm.list_threads(tenant_id="bob")}
    assert bob == {"t-bob"}


def test_list_threads_none_tenant_sees_all(pm):
    pm.upsert_meta("t-alice", title="A", tenant_id="alice")
    pm.upsert_meta("t-bob", title="B", tenant_id="bob")
    everyone = {m.thread_id for m in pm.list_threads(tenant_id=None)}
    assert everyone == {"t-alice", "t-bob"}


def test_get_meta_cross_tenant_returns_none(pm):
    pm.upsert_meta("t-alice", title="A", tenant_id="alice")
    assert pm.get_meta("t-alice", tenant_id="bob") is None
    assert pm.get_meta("t-alice", tenant_id="alice") is not None


def test_find_by_prefix_respects_tenant(pm):
    pm.upsert_meta("abcd-alice", title="A", tenant_id="alice")
    pm.upsert_meta("abcd-bob", title="B", tenant_id="bob")
    hits = {m.thread_id for m in pm.find_by_prefix("abcd", tenant_id="alice")}
    assert hits == {"abcd-alice"}


def test_delete_thread_cross_tenant_refused(pm):
    pm.upsert_meta("t-alice", title="A", tenant_id="alice")
    assert pm.delete_thread("t-alice", tenant_id="bob") is False
    assert pm.get_meta("t-alice") is not None  # 数据未被误删
    assert pm.delete_thread("t-alice", tenant_id="alice") is True
    assert pm.get_meta("t-alice") is None


def test_audit_events_filter_by_tenant(pm):
    pm.log_audit_event("t1", kind="tool_call", tool_name="x", tenant_id="alice")
    pm.log_audit_event("t2", kind="tool_call", tool_name="y", tenant_id="bob")
    alice = pm.list_audit_events(tenant_id="alice")
    assert [e.tool_name for e in alice] == ["x"]


def test_usage_summary_filter_by_tenant(pm):
    pm.log_audit_event("t1", kind="llm_call", model_label="m", input_tokens=10,
                       output_tokens=5, tenant_id="alice")
    pm.log_audit_event("t2", kind="llm_call", model_label="m", input_tokens=99,
                       output_tokens=99, tenant_id="bob")
    summary = pm.usage_summary(tenant_id="alice")
    models = summary["models"]
    assert len(models) == 1
    assert models[0]["input_tokens"] == 10
    assert models[0]["output_tokens"] == 5


# ── UPSERT COALESCE 语义 ─────────────────────────────────────────────────

def test_upsert_title_none_keeps_old_value(pm):
    pm.upsert_meta("t", title="原标题")
    pm.upsert_meta("t", title=None, preview="新预览")  # title=None 不应覆盖
    meta = pm.get_meta("t")
    assert meta.title == "原标题"
    assert meta.preview == "新预览"


def test_upsert_message_count_none_keeps_old_value(pm):
    pm.upsert_meta("t", message_count=5)
    pm.upsert_meta("t", intent="chat")  # message_count 未给 → 保留
    assert pm.get_meta("t").message_count == 5


def test_upsert_tenant_not_stolen_by_later_write(pm):
    """tenant_id 用 COALESCE(旧, 新)：首次归属后不被后写入抢走。"""
    pm.upsert_meta("t", title="A", tenant_id="alice")
    pm.upsert_meta("t", preview="p", tenant_id="bob")
    assert pm.get_meta("t").tenant_id == "alice"


def test_upsert_title_derives_from_preview_on_insert(pm):
    pm.upsert_meta("t", preview="从预览派生的标题内容")
    assert pm.get_meta("t").title == "从预览派生的标题内容"


# ── checkpoint label 增查 ────────────────────────────────────────────────

def test_checkpoint_label_set_and_resolve(pm):
    assert pm.set_checkpoint_label("t", "ckpt-1", "before-edit") is True
    assert pm.resolve_checkpoint_label("t", "before-edit") == "ckpt-1"


def test_checkpoint_label_move_on_conflict(pm):
    pm.set_checkpoint_label("t", "ckpt-1", "milestone")
    pm.set_checkpoint_label("t", "ckpt-2", "milestone")  # 同名 label 移动指向
    assert pm.resolve_checkpoint_label("t", "milestone") == "ckpt-2"
    labels = pm.list_checkpoint_labels("t")
    assert len(labels) == 1


def test_checkpoint_label_rejects_empty(pm):
    assert pm.set_checkpoint_label("t", "ckpt", "") is False
    assert pm.set_checkpoint_label("t", "", "label") is False
    assert pm.resolve_checkpoint_label("t", "missing") is None


@pytest.mark.parametrize("db_env", ["ASKANSWER_DB_PATH"])
def test_default_db_path_honors_env(monkeypatch, tmp_path, db_env):
    from askanswer.persistence import default_db_path

    target = tmp_path / "custom.db"
    monkeypatch.setenv(db_env, str(target))
    assert default_db_path() == target
