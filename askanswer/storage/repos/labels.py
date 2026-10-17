"""checkpoint_label 表：/undo 命名点。"""

from __future__ import annotations

import time

from askanswer.storage.db import Database


class LabelRepo:
    def __init__(self, db: Database):
        self._db = db

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
        with self._db.lock, self._db.conn:
            self._db.conn.execute(
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
        with self._db.lock:
            row = self._db.conn.execute(
                "SELECT checkpoint_id FROM checkpoint_label "
                "WHERE thread_id = ? AND label = ?",
                (thread_id, label),
            ).fetchone()
        return row[0] if row else None

    def list_checkpoint_labels(self, thread_id: str) -> list[dict]:
        """列出某 thread 的所有 label，按创建时间倒序。"""
        with self._db.lock:
            rows = self._db.conn.execute(
                "SELECT label, checkpoint_id, created_at FROM checkpoint_label "
                "WHERE thread_id = ? ORDER BY created_at DESC",
                (thread_id,),
            ).fetchall()
        return [
            {"label": r[0], "checkpoint_id": r[1], "created_at": int(r[2] or 0)}
            for r in rows
        ]
