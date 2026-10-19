"""Thread metadata and checkpoint-label storage (``thread_meta`` / ``checkpoint_label``).

Implemented as a mixin over a shared ``sqlite3`` connection: the host class
(:class:`askanswer.storage.manager.PersistenceManager`) provides ``self._conn``
and ``self._lock``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

from ._util import default_tenant_id as _default_tenant_id


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
    tenant_id: str | None = None  # 归属租户；NULL 视为“不分租户”桶


class ThreadStoreMixin:
    """``thread_meta`` UPSERT / 查询 / 删除，以及 ``checkpoint_label`` CRUD（含租户隔离）。"""

    _conn: sqlite3.Connection

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
        tenant = tenant_id if tenant_id is not None else _default_tenant_id()
        # INSERT 分支的 title 默认值：未显式给出时回退到 preview 派生。
        # UPDATE 分支必须使用原始 title（可能是 None），否则首次派生的标题
        # 会在后续调用上把已存在的好标题覆盖掉。
        insert_title = title if title is not None else _derive_title(preview)
        insert_count = int(message_count or 0)
        update_count = int(message_count) if message_count is not None else None
        with self._lock, self._conn:
            self._conn.execute(
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
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_meta(r) for r in rows]

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
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return _row_to_meta(row) if row else None

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
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
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

    def delete_thread(self, thread_id: str, *, tenant_id: str | None = None) -> bool:
        """同时清掉 SqliteSaver 三张内部表 + thread_meta。

        SqliteSaver 内部表名按版本可能略有差异；用 try/except 兜底，对没有这张表
        或没有 thread_id 列的情况静默跳过。

        ``tenant_id`` 非 None 时先校验归属：越租户删除直接返回 False，且不动任何
        checkpoint 数据（避免 A 租户删掉 B 租户的会话）。
        """
        if tenant_id is not None:
            owner = self.get_meta(thread_id)
            if owner is None or owner.tenant_id != tenant_id:
                return False
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
            try:
                self._conn.execute(
                    "DELETE FROM checkpoint_label WHERE thread_id = ?", (thread_id,)
                )
            except sqlite3.OperationalError:
                pass
            existed = cur.rowcount > 0
        return existed

    # ── checkpoint labels（/undo 命名点） ────────────────────────────────

    def set_checkpoint_label(
        self,
        thread_id: str,
        checkpoint_id: str,
        label: str,
    ) -> bool:
        """给某个 checkpoint 打 label（同 thread 内 label 唯一，重复则移动指向）。

        返回是否写入成功（label / checkpoint_id 任一为空则拒绝，返回 False）。
        """
        label = (label or "").strip()
        if not thread_id or not checkpoint_id or not label:
            return False
        now = int(time.time())
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO checkpoint_label(thread_id, checkpoint_id, label, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id, label) DO UPDATE SET
                    checkpoint_id = excluded.checkpoint_id,
                    created_at    = excluded.created_at
                """,
                (thread_id, checkpoint_id, label, now),
            )
        return True

    def resolve_checkpoint_label(self, thread_id: str, label: str) -> str | None:
        """按 label 反查 checkpoint_id；找不到返回 None。"""
        label = (label or "").strip()
        if not thread_id or not label:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT checkpoint_id FROM checkpoint_label "
                "WHERE thread_id = ? AND label = ?",
                (thread_id, label),
            ).fetchone()
        return row[0] if row else None

    def list_checkpoint_labels(self, thread_id: str) -> list[dict]:
        """列出某 thread 的所有 label，按创建时间倒序。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT label, checkpoint_id, created_at FROM checkpoint_label "
                "WHERE thread_id = ? ORDER BY created_at DESC",
                (thread_id,),
            ).fetchall()
        return [
            {"label": r[0], "checkpoint_id": r[1], "created_at": int(r[2] or 0)}
            for r in rows
        ]


# ── helpers ───────────────────────────────────────────────────────────

def _derive_title(preview: str | None) -> str | None:
    """从 preview 抽取一行作为默认 title（首次插入时使用）。"""
    if not preview:
        return None
    line = preview.strip().splitlines()[0] if preview.strip() else ""
    return line[:30] or None


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
        tenant_id,
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
        tenant_id=tenant_id,
    )
