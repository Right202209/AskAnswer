# 持久化与线程元数据管理（CLI 单用户场景）。
#
# 设计要点：
# - 一个 SQLite 文件同时承载两件事：
#   (a) LangGraph 自带的 ``SqliteSaver`` checkpoint 表（``checkpoints`` /
#       ``writes`` / ``checkpoint_blobs``，由 langgraph-checkpoint-sqlite 自管）；
#   (b) 我们自己的 ``thread_meta`` 表（title / preview / 时间戳 / 计数）。
#   两者共享同一连接，避免跨 attached DB 的事务复杂度。
# - 多进程并发：开 WAL + busy_timeout，让两个终端同时跑 askanswer 不会
#   立刻 ``database is locked``。
# - 模式迁移：用一张独立的 ``askanswer_schema`` 元表记录我们自己的版本号，
#   不依赖 ``PRAGMA user_version``（避免与 SqliteSaver 未来潜在用法冲突）。
# - ``--graph`` 只导出 Mermaid 不应触碰持久化：因此本模块完全 lazy，
#   ``get_persistence()`` 是显式调用入口；``graph.py`` 在导出图时传
#   ``InMemorySaver`` 绕过本模块。
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver


# 当前自管 schema 版本号；新增列 / 表时把版本号 +1，并在 ``_migrate`` 里追加迁移分支。
_SCHEMA_VERSION = 2


def default_db_path() -> Path:
    """返回默认的 state.db 路径。

    优先级：``ASKANSWER_DB_PATH`` > ``XDG_DATA_HOME/askanswer`` > ``~/.askanswer``。
    """
    explicit = os.environ.get("ASKANSWER_DB_PATH")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() / "askanswer" if xdg else Path.home() / ".askanswer"
    return base / "state.db"


@dataclass
class ThreadMeta:
    """``thread_meta`` 表的 Python 视图。"""

    thread_id: str
    title: str | None
    tags: list[str] = field(default_factory=list)
    created_at: int = 0          # epoch seconds
    updated_at: int = 0          # epoch seconds
    message_count: int = 0       # 用户回合数（HumanMessage 数）
    last_intent: str | None = None
    model_label: str | None = None
    preview: str | None = None   # 最近一条用户消息前 80 字符


@dataclass
class AuditEvent:
    """``audit_event`` 表的一行。"""

    id: int
    thread_id: str
    ts: int
    kind: str
    tool_name: str | None = None
    args_summary: str | None = None
    result_size: int | None = None
    model_label: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int | None = None
    intent: str | None = None
    error: str | None = None


class PersistenceManager:
    """SqliteSaver + thread_meta 的统一管理者。

    并发说明：
    - 同进程内，多线程通过 ``check_same_thread=False`` 共用一个连接；
      SQLite 自身在写入时会序列化，我们额外用 ``self._lock`` 保护
      “SELECT 后 UPDATE / DELETE” 这类需要原子性的复合操作。
    - 多进程通过 WAL + ``busy_timeout`` 容忍并发，仍可能在极端情况下
      出现 ``OperationalError: database is locked``；上层捕获后重试即可。
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = Lock()
        # 父目录可能不存在（首次运行）
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：LangGraph 节点会在 worker 线程上执行 saver 写入；
        # isolation_level="" 保留 Python sqlite3 默认的 "BEGIN on first DML" 行为，
        # 这样 ``with self._conn`` 才能正确开/提交事务。
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level="",
        )
        # 多进程并发的关键三件套：WAL 让读不阻塞写、busy_timeout 让 SQLite
        # 内部自旋等待而不是立刻报错、synchronous=NORMAL 在 WAL 下足够安全。
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # SqliteSaver 共用同一连接：``setup()`` 是 idempotent 的 CREATE IF NOT EXISTS
        self._saver = SqliteSaver(self._conn)
        self._saver.setup()

        # 我们自己的迁移
        self._migrate()

    # ── public surface ────────────────────────────────────────────────
    @property
    def checkpointer(self) -> SqliteSaver:
        return self._saver

    @property
    def db_path(self) -> Path:
        return self._db_path

    def upsert_meta(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        intent: str | None = None,
        model_label: str | None = None,
        preview: str | None = None,
        message_count: int | None = None,
    ) -> None:
        """写入或更新一行 thread_meta。

        语义：
        - title：仅在首次插入时若未显式给出，则从 preview 截 30 字符兜底；
          后续调用若 title=None 则保留旧值（不被覆盖）。
        - 其它字段：None 表示“不动”，非 None 表示“覆盖”。
        - updated_at 每次都刷新到当前时间。
        """
        now = int(time.time())
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT 1 FROM thread_meta WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is None:
                final_title = title or _derive_title(preview)
                self._conn.execute(
                    """
                    INSERT INTO thread_meta(
                        thread_id, title, tags, created_at, updated_at,
                        message_count, last_intent, model_label, preview
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        final_title,
                        json.dumps([]),
                        now,
                        now,
                        int(message_count or 0),
                        intent,
                        model_label,
                        preview,
                    ),
                )
            else:
                # COALESCE(?, col)：传入 NULL 时保留旧值，非 NULL 时覆盖
                self._conn.execute(
                    """
                    UPDATE thread_meta
                    SET title         = COALESCE(?, title),
                        last_intent   = COALESCE(?, last_intent),
                        model_label   = COALESCE(?, model_label),
                        preview       = COALESCE(?, preview),
                        message_count = COALESCE(?, message_count),
                        updated_at    = ?
                    WHERE thread_id = ?
                    """,
                    (
                        title,
                        intent,
                        model_label,
                        preview,
                        message_count,
                        now,
                        thread_id,
                    ),
                )

    def list_threads(
        self,
        limit: int = 50,
        query: str | None = None,
    ) -> list[ThreadMeta]:
        """按 updated_at 倒序列出，可选按 title/preview 模糊匹配。"""
        sql = (
            "SELECT thread_id, title, tags, created_at, updated_at, "
            "message_count, last_intent, model_label, preview "
            "FROM thread_meta "
        )
        params: list = []
        q = (query or "").strip()
        if q:
            sql += "WHERE title LIKE ? OR preview LIKE ? "
            like = f"%{q}%"
            params.extend([like, like])
        sql += "ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_meta(r) for r in rows]

    def get_meta(self, thread_id: str) -> ThreadMeta | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT thread_id, title, tags, created_at, updated_at, "
                "message_count, last_intent, model_label, preview "
                "FROM thread_meta WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return _row_to_meta(row) if row else None

    def find_by_prefix(self, prefix: str, limit: int = 5) -> list[ThreadMeta]:
        """按 thread_id 前缀匹配（4 字符及以上才生效）。"""
        prefix = (prefix or "").strip()
        if len(prefix) < 4:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT thread_id, title, tags, created_at, updated_at, "
                "message_count, last_intent, model_label, preview "
                "FROM thread_meta "
                "WHERE thread_id LIKE ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (f"{prefix}%", int(limit)),
            ).fetchall()
        return [_row_to_meta(r) for r in rows]

    def set_title(self, thread_id: str, title: str) -> bool:
        now = int(time.time())
        title = (title or "").strip()
        if not title:
            return False
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE thread_meta SET title = ?, updated_at = ? WHERE thread_id = ?",
                (title, now, thread_id),
            )
            return cur.rowcount > 0

    def delete_thread(self, thread_id: str) -> bool:
        """同时清掉 SqliteSaver 三张内部表 + thread_meta。

        SqliteSaver 内部表名按版本可能略有差异；用 try/except 兜底，对没有这张表
        或没有 thread_id 列的情况静默跳过。
        """
        existed = False
        with self._lock, self._conn:
            # SqliteSaver 内部表：v2 是 checkpoints / writes / checkpoint_blobs
            for table in ("writes", "checkpoint_blobs", "checkpoints"):
                try:
                    self._conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,)
                    )
                except sqlite3.OperationalError:
                    # 表或列不存在：版本差异导致，忽略
                    continue
            cur = self._conn.execute(
                "DELETE FROM thread_meta WHERE thread_id = ?", (thread_id,)
            )
            try:
                self._conn.execute(
                    "DELETE FROM audit_event WHERE thread_id = ?", (thread_id,)
                )
            except sqlite3.OperationalError:
                pass
            existed = cur.rowcount > 0
        return existed

    def log_audit_event(
        self,
        thread_id: str,
        *,
        kind: str,
        ts: int | None = None,
        tool_name: str | None = None,
        args_summary: str | None = None,
        result_size: int | None = None,
        model_label: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        duration_ms: int | None = None,
        intent: str | None = None,
        error: str | None = None,
    ) -> None:
        """追加一条审计事件。"""
        if not thread_id or not kind:
            return
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO audit_event(
                    thread_id, ts, kind, tool_name, args_summary, result_size,
                    model_label, input_tokens, output_tokens, duration_ms,
                    intent, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    int(ts or time.time()),
                    kind,
                    tool_name,
                    _clip(args_summary, 200) if args_summary is not None else None,
                    result_size,
                    model_label,
                    input_tokens,
                    output_tokens,
                    duration_ms,
                    intent,
                    _clip(error, 200) if error is not None else None,
                ),
            )

    def list_audit_events(
        self,
        *,
        thread_id: str | None = None,
        limit: int = 50,
        kind: str | None = None,
        days: int | None = None,
    ) -> list[AuditEvent]:
        """按时间倒序列出审计事件。"""
        sql = (
            "SELECT id, thread_id, ts, kind, tool_name, args_summary, result_size, "
            "model_label, input_tokens, output_tokens, duration_ms, intent, error "
            "FROM audit_event "
        )
        where: list[str] = []
        params: list = []
        if thread_id:
            where.append("thread_id = ?")
            params.append(thread_id)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if days and days > 0:
            where.append("ts >= ?")
            params.append(int(time.time()) - int(days) * 86400)
        if where:
            sql += "WHERE " + " AND ".join(where) + " "
        sql += "ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_audit(r) for r in rows]

    def usage_summary(
        self,
        *,
        thread_id: str | None = None,
        days: int | None = None,
    ) -> dict[str, list[dict]]:
        """返回模型与工具两个维度的轻量聚合。"""
        where: list[str] = []
        params: list = []
        if thread_id:
            where.append("thread_id = ?")
            params.append(thread_id)
        if days and days > 0:
            where.append("ts >= ?")
            params.append(int(time.time()) - int(days) * 86400)
        clause = "WHERE " + " AND ".join(where) if where else ""

        with self._lock:
            model_rows = self._conn.execute(
                f"""
                SELECT COALESCE(model_label, ''), COUNT(*),
                       COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(duration_ms), 0)
                FROM audit_event
                {clause}
                {"AND" if where else "WHERE"} kind = 'llm_call'
                GROUP BY COALESCE(model_label, '')
                ORDER BY COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(output_tokens), 0) DESC
                """,
                params,
            ).fetchall()
            tool_rows = self._conn.execute(
                f"""
                SELECT COALESCE(tool_name, kind), COUNT(*),
                       COALESCE(SUM(result_size), 0),
                       COALESCE(SUM(duration_ms), 0),
                       COALESCE(SUM(CASE WHEN error IS NULL OR error = '' THEN 0 ELSE 1 END), 0)
                FROM audit_event
                {clause}
                {"AND" if where else "WHERE"} kind IN ('tool_call', 'shell_approve', 'shell_reject', 'mcp_connect', 'model_swap')
                GROUP BY COALESCE(tool_name, kind)
                ORDER BY 2 DESC
                """,
                params,
            ).fetchall()

        return {
            "models": [
                {
                    "model_label": row[0] or None,
                    "calls": int(row[1] or 0),
                    "input_tokens": int(row[2] or 0),
                    "output_tokens": int(row[3] or 0),
                    "duration_ms": int(row[4] or 0),
                }
                for row in model_rows
            ],
            "tools": [
                {
                    "name": row[0] or None,
                    "calls": int(row[1] or 0),
                    "result_size": int(row[2] or 0),
                    "duration_ms": int(row[3] or 0),
                    "errors": int(row[4] or 0),
                }
                for row in tool_rows
            ],
        }

    def import_audit_events(
        self,
        events: list[dict],
        *,
        thread_id: str,
    ) -> int:
        """把导出的审计事件恢复到新 thread_id 下，返回写入数量。"""
        count = 0
        for event in events:
            if not isinstance(event, dict):
                continue
            self.log_audit_event(
                thread_id,
                ts=_int_or_none(event.get("ts")) or int(time.time()),
                kind=str(event.get("kind") or "imported"),
                tool_name=event.get("tool_name"),
                args_summary=event.get("args_summary"),
                result_size=_int_or_none(event.get("result_size")),
                model_label=event.get("model_label"),
                input_tokens=_int_or_none(event.get("input_tokens")),
                output_tokens=_int_or_none(event.get("output_tokens")),
                duration_ms=_int_or_none(event.get("duration_ms")),
                intent=event.get("intent"),
                error=event.get("error"),
            )
            count += 1
        return count

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # ── schema migration ──────────────────────────────────────────────
    def _migrate(self) -> None:
        """创建 / 升级 ``askanswer_schema`` 与 ``thread_meta``。"""
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS askanswer_schema (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            row = self._conn.execute(
                "SELECT value FROM askanswer_schema WHERE key = 'version'"
            ).fetchone()
            current = int(row[0]) if row and row[0] is not None else 0

            if current < 1:
                self._conn.execute(
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
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_thread_meta_updated "
                    "ON thread_meta(updated_at DESC)"
                )
                current = 1

            if current < 2:
                self._conn.execute(
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
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_thread_ts "
                    "ON audit_event(thread_id, ts DESC)"
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_kind_ts "
                    "ON audit_event(kind, ts DESC)"
                )
                current = 2

            # 未来加列 / 加表的迁移分支模板：
            # if current < 2:
            #     self._conn.execute("ALTER TABLE thread_meta ADD COLUMN summary TEXT")
            #     current = 2

            self._conn.execute(
                "INSERT OR REPLACE INTO askanswer_schema(key, value) VALUES('version', ?)",
                (str(current),),
            )


# ── helpers ───────────────────────────────────────────────────────────

def _derive_title(preview: str | None) -> str | None:
    """从 preview 抽取一行作为默认 title（首次插入时使用）。"""
    if not preview:
        return None
    line = preview.strip().splitlines()[0] if preview.strip() else ""
    return line[:30] or None


def _clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _int_or_none(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_to_meta(row) -> ThreadMeta:
    """把 SELECT 出来的 row 还原为 ``ThreadMeta``。"""
    (
        thread_id,
        title,
        tags_json,
        created_at,
        updated_at,
        message_count,
        last_intent,
        model_label,
        preview,
    ) = row
    try:
        tags = json.loads(tags_json) if tags_json else []
    except json.JSONDecodeError:
        tags = []
    if not isinstance(tags, list):
        tags = []
    return ThreadMeta(
        thread_id=thread_id,
        title=title,
        tags=tags,
        created_at=int(created_at or 0),
        updated_at=int(updated_at or 0),
        message_count=int(message_count or 0),
        last_intent=last_intent,
        model_label=model_label,
        preview=preview,
    )


def _row_to_audit(row) -> AuditEvent:
    (
        event_id,
        thread_id,
        ts,
        kind,
        tool_name,
        args_summary,
        result_size,
        model_label,
        input_tokens,
        output_tokens,
        duration_ms,
        intent,
        error,
    ) = row
    return AuditEvent(
        id=int(event_id or 0),
        thread_id=thread_id,
        ts=int(ts or 0),
        kind=kind,
        tool_name=tool_name,
        args_summary=args_summary,
        result_size=_int_or_none(result_size),
        model_label=model_label,
        input_tokens=_int_or_none(input_tokens),
        output_tokens=_int_or_none(output_tokens),
        duration_ms=_int_or_none(duration_ms),
        intent=intent,
        error=error,
    )


# ── singleton ─────────────────────────────────────────────────────────

_singleton: Optional[PersistenceManager] = None
_singleton_lock = Lock()


def get_persistence(db_path: Path | None = None) -> PersistenceManager:
    """返回进程内 PersistenceManager 单例；首次调用时按需打开 SQLite。"""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = PersistenceManager(db_path or default_db_path())
        return _singleton


def shutdown_persistence() -> None:
    """关闭单例（``atexit`` 注册用）。重复调用 idempotent。"""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.close()
            _singleton = None
