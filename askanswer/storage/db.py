"""``Database``：SQLite 连接、并发设置与自管 schema 迁移（架构文档 6.13.2）。

一个 SQLite 文件同时承载 LangGraph ``SqliteSaver`` 的 checkpoint 表与我们自己的
``thread_meta`` / ``audit_event`` / ``checkpoint_label`` 表，共用同一连接。
多进程并发：WAL + busy_timeout + synchronous=NORMAL。
迁移用独立的 ``askanswer_schema`` 元表记版本，不依赖 ``PRAGMA user_version``。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock

# 当前自管 schema 版本号；新增列 / 表时 +1，并在 ``_migrate`` 里追加迁移分支。
SCHEMA_VERSION = 4


def default_db_path() -> Path:
    """默认 state.db 路径：``ASKANSWER_DB_PATH`` > ``XDG_DATA_HOME/askanswer`` > ``~/.askanswer``。"""
    from askanswer.config import current_settings
    from askanswer.config.paths import db_path

    return db_path(current_settings())


def default_tenant_id() -> str | None:
    """写入路径未显式给 tenant 时的兜底（Settings.tenant_id，缺省 None = 不分租户）。"""
    from askanswer.config import current_settings

    return current_settings().tenant_id


class Database:
    """连接 + 锁 + 迁移。repos 共享同一个 Database 实例。

    同进程多线程通过 ``check_same_thread=False`` 共用连接；``lock`` 保护
    「SELECT 后 UPDATE / DELETE」这类复合操作。
    """

    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level="" 保留 "BEGIN on first DML"，``with conn`` 才能正确开/提交事务
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level="")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA synchronous=NORMAL")

    def migrate(self) -> None:
        self._migrate()

    def close(self) -> None:
        with self.lock:
            try:
                self.conn.close()
            except Exception:
                pass

    def _migrate(self) -> None:
        """创建 / 升级 ``askanswer_schema`` 与 ``thread_meta``。"""
        with self.lock, self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS askanswer_schema (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            row = self.conn.execute(
                "SELECT value FROM askanswer_schema WHERE key = 'version'"
            ).fetchone()
            current = int(row[0]) if row and row[0] is not None else 0

            if current < 1:
                self.conn.execute(
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
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_thread_meta_updated "
                    "ON thread_meta(updated_at DESC)"
                )
                current = 1

            if current < 2:
                self.conn.execute(
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
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_thread_ts "
                    "ON audit_event(thread_id, ts DESC)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_kind_ts "
                    "ON audit_event(kind, ts DESC)"
                )
                current = 2

            if current < 3:
                # 多租户隔离：给两张表加 tenant_id 列（旧行迁移后为 NULL，行为不退化），
                # 并建按 tenant 过滤的复合索引。SqliteSaver 内部表不动，隔离在
                # thread_meta / audit_event 层做（见 execution-plan Phase 1.3）。
                self._add_column_if_missing("thread_meta", "tenant_id", "TEXT")
                self._add_column_if_missing("audit_event", "tenant_id", "TEXT")
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_thread_meta_tenant_updated "
                    "ON thread_meta(tenant_id, updated_at DESC)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_tenant_ts "
                    "ON audit_event(tenant_id, ts DESC)"
                )
                current = 3

            if current < 4:
                # /undo 命名点：给 checkpoint 打 label，让用户回到有意义的还原点。
                # (thread_id, label) 作主键：同名 label 再次打标即“移动”到新 checkpoint。
                self.conn.execute(
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
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_checkpoint_label_thread "
                    "ON checkpoint_label(thread_id, created_at DESC)"
                )
                current = 4

            self.conn.execute(
                "INSERT OR REPLACE INTO askanswer_schema(key, value) VALUES('version', ?)",
                (str(current),),
            )

    def _add_column_if_missing(self, table: str, column: str, decl: str) -> None:
        """幂等地给已有表加列：列已存在时静默跳过。

        ``ALTER TABLE ADD COLUMN`` 在列重复时会抛 OperationalError；用 PRAGMA
        先查一遍现有列，避免依赖异常字符串匹配。
        """
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(row[1]) for row in rows}
        if column in existing:
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
