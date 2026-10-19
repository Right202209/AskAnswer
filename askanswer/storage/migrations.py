"""Schema migrations for AskAnswer's own SQLite tables.

The version lives in a dedicated ``askanswer_schema`` table rather than
``PRAGMA user_version`` so it never collides with ``SqliteSaver``.  Bump
:data:`SCHEMA_VERSION` and append a branch to :func:`migrate` for every change.

- v1: ``thread_meta``
- v2: ``audit_event``
- v3: ``tenant_id`` columns + tenant indexes (multi-tenant isolation)
- v4: ``checkpoint_label`` (named restore points)
- v5: ``audit_event.cached_tokens`` (prompt-cache hits reported by the provider)
"""

from __future__ import annotations

import sqlite3

# 当前自管 schema 版本号；新增列 / 表时把版本号 +1，并在 ``migrate`` 里追加迁移分支。
SCHEMA_VERSION = 5


def schema_version(conn: sqlite3.Connection) -> int:
    """Return the recorded schema version (0 when the meta table is missing)."""
    try:
        row = conn.execute(
            "SELECT value FROM askanswer_schema WHERE key = 'version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row and row[0] is not None else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Create / upgrade our tables in place and return the resulting version.

    The caller owns locking and the transaction (``with conn:``).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS askanswer_schema (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    current = schema_version(conn)

    if current < 1:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_meta (
                thread_id TEXT PRIMARY KEY,
                title TEXT,
                tags TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                last_intent TEXT,
                model_label TEXT,
                preview TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_thread_meta_updated "
            "ON thread_meta(updated_at DESC)"
        )
        current = 1

    if current < 2:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_event (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                ts INTEGER NOT NULL,
                kind TEXT NOT NULL,
                tool_name TEXT,
                args_summary TEXT,
                result_size INTEGER,
                model_label TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                duration_ms INTEGER,
                intent TEXT,
                error TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_thread_ts "
            "ON audit_event(thread_id, ts DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_kind_ts "
            "ON audit_event(kind, ts DESC)"
        )
        current = 2

    if current < 3:
        # 多租户隔离：给两张表加 tenant_id 列（旧行迁移后为 NULL，行为不退化），
        # 并建按 tenant 过滤的复合索引。SqliteSaver 内部表不动，隔离在
        # thread_meta / audit_event 层做。
        add_column_if_missing(conn, "thread_meta", "tenant_id", "TEXT")
        add_column_if_missing(conn, "audit_event", "tenant_id", "TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_thread_meta_tenant_updated "
            "ON thread_meta(tenant_id, updated_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_tenant_ts "
            "ON audit_event(tenant_id, ts DESC)"
        )
        current = 3

    if current < 4:
        # /undo 命名点：给 checkpoint 打 label，让用户回到有意义的还原点。
        # (thread_id, label) 作主键：同名 label 再次打标即“移动”到新 checkpoint。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_label (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                label TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (thread_id, label)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkpoint_label_thread "
            "ON checkpoint_label(thread_id, created_at DESC)"
        )
        current = 4

    if current < 5:
        # 缓存命中 token：LLMUsageCallback 早已在内存里采集 cached_tokens，v5 起落库。
        # 旧行默认 0（"未知 / 无缓存"），聚合时不需要 COALESCE 特判。
        add_column_if_missing(conn, "audit_event", "cached_tokens", "INTEGER DEFAULT 0")
        current = 5

    conn.execute(
        "INSERT OR REPLACE INTO askanswer_schema(key, value) VALUES('version', ?)",
        (str(current),),
    )
    return current


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """幂等地给已有表加列：列已存在时静默跳过。

    ``ALTER TABLE ADD COLUMN`` 在列重复时会抛 OperationalError；用 PRAGMA
    先查一遍现有列，避免依赖异常字符串匹配。
    """
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row[1]) for row in rows}
    if column in existing:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
