"""thread_meta 表的读写（架构文档 6.13.3）。"""

from __future__ import annotations

import json
import sqlite3
import time

from askanswer.storage.db import Database, default_tenant_id
from askanswer.storage.models import ThreadMeta, derive_title, row_to_meta


class ThreadRepo:
    def __init__(self, db: Database):
        self._db = db

    def upsert_meta(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        intent: str | None = None,
        model_label: str | None = None,
        preview: str | None = None,
        message_count: int | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """写入或更新一行 thread_meta。

        语义：
        - title：仅在首次插入时若未显式给出，则从 preview 截 30 字符兜底；
          后续调用若 title=None 则保留旧值（不被覆盖）。
        - 其它字段：None 表示“不动”，非 None 表示“覆盖”。
        - updated_at 取 max(now, 旧值)，避免跨进程时钟回退导致排序异常。
        - created_at 仅 INSERT 阶段写入；UPDATE 分支不动它。
        - tenant_id：写入路径的“归属租户”。未显式传入则回退到
          ``ASKANSWER_TENANT_ID`` 环境变量，再兜底 None（不分租户）。
          UPDATE 分支用 COALESCE 保留既有归属，只在原值为 NULL 时才被采纳，
          避免线程被后写入的租户“抢走”。

        以单条 SQLite UPSERT 完成（``INSERT … ON CONFLICT … DO UPDATE``），
        消除了 “SELECT 1 → INSERT/UPDATE” 的逻辑竞争窗口：多进程同 thread
        高频写入时不会丢 title/preview 或抛 UNIQUE 约束错误。
        ``title`` 和 ``message_count`` 在 INSERT 和 UPDATE 两条分支语义不同
        （INSERT 走默认值；UPDATE 走 “None 即保留”），因此各自单独绑定。
        """
        now = int(time.time())
        tenant = tenant_id if tenant_id is not None else default_tenant_id()
        # INSERT 分支的 title 默认值：未显式给出时回退到 preview 派生。
        # UPDATE 分支必须使用原始 title（可能是 None），否则首次派生的标题
        # 会在后续调用上把已存在的好标题覆盖掉。
        insert_title = title if title is not None else derive_title(preview)
        insert_count = int(message_count or 0)
        update_count = int(message_count) if message_count is not None else None
        with self._db.lock, self._db.conn:
            self._db.conn.execute(
                """
                INSERT INTO thread_meta(
                    thread_id, title, tags, created_at, updated_at,
                    message_count, last_intent, model_label, preview, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    title         = COALESCE(?, thread_meta.title),
                    last_intent   = COALESCE(excluded.last_intent, thread_meta.last_intent),
                    model_label   = COALESCE(excluded.model_label, thread_meta.model_label),
                    preview       = COALESCE(excluded.preview, thread_meta.preview),
                    message_count = COALESCE(?, thread_meta.message_count),
                    tenant_id     = COALESCE(thread_meta.tenant_id, excluded.tenant_id),
                    updated_at    = MAX(excluded.updated_at, thread_meta.updated_at)
                """,
                (
                    thread_id,
                    insert_title,
                    json.dumps([]),
                    now,
                    now,
                    insert_count,
                    intent,
                    model_label,
                    preview,
                    tenant,
                    # DO UPDATE 分支专用绑定：raw title / raw message_count
                    title,
                    update_count,
                ),
            )

    def list_threads(
        self,
        limit: int = 50,
        query: str | None = None,
        *,
        tenant_id: str | None = None,
    ) -> list[ThreadMeta]:
        """按 updated_at 倒序列出，可选按 title/preview 模糊匹配。

        ``tenant_id`` 为 None 时不做租户过滤（等价旧行为，兼容既有调用）；
        非 None 时只返回该租户的线程（``tenant_id`` 为 NULL 的旧数据不会串到
        某个具名租户下）。
        """
        sql = (
            "SELECT thread_id, title, tags, created_at, updated_at, "
            "message_count, last_intent, model_label, preview, tenant_id "
            "FROM thread_meta "
        )
        params: list = []
        where: list[str] = []
        q = (query or "").strip()
        if q:
            where.append("(title LIKE ? OR preview LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like])
        if tenant_id is not None:
            where.append("tenant_id = ?")
            params.append(tenant_id)
        if where:
            sql += "WHERE " + " AND ".join(where) + " "
        sql += "ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        with self._db.lock:
            rows = self._db.conn.execute(sql, params).fetchall()
        return [row_to_meta(r) for r in rows]

    def get_meta(self, thread_id: str, *, tenant_id: str | None = None) -> ThreadMeta | None:
        """取单条线程元数据；``tenant_id`` 非 None 时越租户访问返回 None。"""
        sql = (
            "SELECT thread_id, title, tags, created_at, updated_at, "
            "message_count, last_intent, model_label, preview, tenant_id "
            "FROM thread_meta WHERE thread_id = ?"
        )
        params: list = [thread_id]
        if tenant_id is not None:
            sql += " AND tenant_id = ?"
            params.append(tenant_id)
        with self._db.lock:
            row = self._db.conn.execute(sql, params).fetchone()
        return row_to_meta(row) if row else None

    def find_by_prefix(
        self,
        prefix: str,
        limit: int = 5,
        *,
        tenant_id: str | None = None,
    ) -> list[ThreadMeta]:
        """按 thread_id 前缀匹配（4 字符及以上才生效）。"""
        prefix = (prefix or "").strip()
        if len(prefix) < 4:
            return []
        sql = (
            "SELECT thread_id, title, tags, created_at, updated_at, "
            "message_count, last_intent, model_label, preview, tenant_id "
            "FROM thread_meta WHERE thread_id LIKE ? "
        )
        params: list = [f"{prefix}%"]
        if tenant_id is not None:
            sql += "AND tenant_id = ? "
            params.append(tenant_id)
        sql += "ORDER BY updated_at DESC LIMIT ?"
        params.append(int(limit))
        with self._db.lock:
            rows = self._db.conn.execute(sql, params).fetchall()
        return [row_to_meta(r) for r in rows]

    def set_title(self, thread_id: str, title: str) -> bool:
        now = int(time.time())
        title = (title or "").strip()
        if not title:
            return False
        with self._db.lock, self._db.conn:
            cur = self._db.conn.execute(
                "UPDATE thread_meta SET title = ?, updated_at = ? WHERE thread_id = ?",
                (title, now, thread_id),
            )
            return cur.rowcount > 0

    def delete_meta(self, thread_id: str) -> bool:
        """删除自有表里该 thread 的元数据、审计与标签；调用方持锁、负责事务。"""
        cur = self._db.conn.execute("DELETE FROM thread_meta WHERE thread_id = ?", (thread_id,))
        existed = cur.rowcount > 0
        for table in ("audit_event", "checkpoint_label"):
            try:
                self._db.conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError:
                continue  # 自有表尚未建（老库 / 功能未启用）
        return existed
