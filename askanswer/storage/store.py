"""``Store``：存储层门面 = Database + 各 repo + SqliteSaver（架构文档 6.13.4）。

上层（CLI 命令、runner、HTTP server）只和 Store 打交道；跨表操作（如删除会话要
同时清 checkpoint 和自有表）在这里协调，repo 各管一张表。
本模块完全 lazy：``get_store()`` 才会打开 SQLite；``--graph`` 导出走 InMemorySaver。
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from langgraph.checkpoint.sqlite import SqliteSaver

from askanswer.storage.db import Database, default_db_path
from askanswer.storage.models import AuditEvent, ThreadMeta
from askanswer.storage.repos import AuditRepo, LabelRepo, ThreadRepo

__all__ = ["AuditEvent", "Store", "ThreadMeta", "default_db_path", "get_store", "shutdown_store"]


class Store:
    def __init__(self, db_path: Path):
        self.db = Database(db_path)
        # SqliteSaver 共用同一连接：setup() 是幂等的 CREATE IF NOT EXISTS
        self._saver = SqliteSaver(self.db.conn)
        self._saver.setup()
        self.db.migrate()
        self.threads = ThreadRepo(self.db)
        self.labels = LabelRepo(self.db)
        self.audit = AuditRepo(self.db)

    # ── 基础 ──────────────────────────────────────────────────────────
    @property
    def checkpointer(self) -> SqliteSaver:
        return self._saver

    @property
    def db_path(self) -> Path:
        return self.db.path

    def close(self) -> None:
        self.db.close()

    # ── 跨表操作 ──────────────────────────────────────────────────────
    def delete_thread(self, thread_id: str, *, tenant_id: str | None = None) -> bool:
        """删除会话：checkpoint 走 SqliteSaver 官方 ``delete_thread``，再清自有表。

        不按表名直接删 checkpointer 内部表（P11）。``tenant_id`` 非 None 时先校验
        归属，越租户删除返回 False 且不动任何数据。
        """
        if tenant_id is not None:
            owner = self.threads.get_meta(thread_id)
            if owner is None or owner.tenant_id != tenant_id:
                return False
        with self.db.lock:
            self._saver.delete_thread(thread_id)
            with self.db.conn:
                return self.threads.delete_meta(thread_id)

    # ── 线程元数据 ────────────────────────────────────────────────────
    def upsert_meta(self, thread_id: str, **kwargs) -> None:
        self.threads.upsert_meta(thread_id, **kwargs)

    def list_threads(self, limit: int = 50, query: str | None = None, *, tenant_id=None):
        return self.threads.list_threads(limit, query, tenant_id=tenant_id)

    def get_meta(self, thread_id: str, *, tenant_id: str | None = None) -> ThreadMeta | None:
        return self.threads.get_meta(thread_id, tenant_id=tenant_id)

    def find_by_prefix(self, prefix: str, limit: int = 5, *, tenant_id=None):
        return self.threads.find_by_prefix(prefix, limit, tenant_id=tenant_id)

    def set_title(self, thread_id: str, title: str) -> bool:
        return self.threads.set_title(thread_id, title)

    # ── checkpoint 标签 ───────────────────────────────────────────────
    def set_checkpoint_label(self, thread_id: str, checkpoint_id: str, label: str) -> bool:
        return self.labels.set_checkpoint_label(thread_id, checkpoint_id, label)

    def resolve_checkpoint_label(self, thread_id: str, label: str) -> str | None:
        return self.labels.resolve_checkpoint_label(thread_id, label)

    def list_checkpoint_labels(self, thread_id: str) -> list[dict]:
        return self.labels.list_checkpoint_labels(thread_id)

    # ── 审计 ──────────────────────────────────────────────────────────
    def log_audit_event(self, thread_id: str, **kwargs) -> None:
        self.audit.log_audit_event(thread_id, **kwargs)

    def log_audit_events(self, events: list[dict]) -> int:
        return self.audit.log_audit_events(events)

    def list_audit_events(self, **kwargs) -> list[AuditEvent]:
        return self.audit.list_audit_events(**kwargs)

    def usage_summary(self, **kwargs) -> dict[str, list[dict]]:
        return self.audit.usage_summary(**kwargs)

    def import_audit_events(self, events: list[dict], *, thread_id: str, tenant_id=None) -> int:
        return self.audit.import_audit_events(events, thread_id=thread_id, tenant_id=tenant_id)


_singleton: Store | None = None
_singleton_lock = Lock()


def get_store(db_path: Path | None = None) -> Store:
    """进程默认 Store；首次调用时按需打开 SQLite。组装根经它注入。"""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = Store(db_path or default_db_path())
        return _singleton


def shutdown_store() -> None:
    """关闭默认 Store（``atexit`` 用）；重复调用幂等。"""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.close()
            _singleton = None
