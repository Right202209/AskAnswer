"""SQLite connection lifecycle, ``SqliteSaver`` wiring and the process singleton.

设计要点：

- 一个 SQLite 文件同时承载 LangGraph 自带的 ``SqliteSaver`` checkpoint 表与我们自己的
  ``thread_meta`` / ``audit_event`` / ``checkpoint_label`` 表；两者共享同一连接，避免跨
  attached DB 的事务复杂度。
- 多进程并发：WAL + ``busy_timeout``，降低并发运行时的锁冲突。
- 完全 lazy：``get_persistence()`` 是显式入口；宿主也可以向 ``create_search_assistant``
  注入其它 checkpointer 绕过本模块。
"""

from __future__ import annotations

import os
import sqlite3
import weakref
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Protocol, runtime_checkable

from langgraph.checkpoint.sqlite import SqliteSaver

from .audit_store import AuditStoreMixin
from .migrations import migrate
from .threads import ThreadStoreMixin


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


@runtime_checkable
class PersistenceBackend(Protocol):
    """The write surface :func:`askanswer.run_leg` needs from a persistence backend.

    :class:`PersistenceManager` (SQLite) and :class:`NullPersistence` (no-op) both
    satisfy it; hosts may pass their own implementation to ``run_leg(persistence=...)``.
    """

    def upsert_meta(self, thread_id: str, **fields: Any) -> None: ...

    def log_audit_event(self, thread_id: str, **fields: Any) -> None: ...

    def log_audit_events(self, events: list[dict]) -> int: ...


class NullPersistence:
    """No-op backend: accepts every write and stores nothing.

    ``run_leg`` falls back to it when the compiled app's checkpointer is not backed
    by a :class:`PersistenceManager` (e.g. ``InMemorySaver``), so in-memory runs
    never create ``state.db``.
    """

    checkpointer = None

    def upsert_meta(self, thread_id: str, **fields: Any) -> None:
        return None

    def log_audit_event(self, thread_id: str, **fields: Any) -> None:
        return None

    def log_audit_events(self, events: list[dict]) -> int:
        return 0

    def __repr__(self) -> str:
        return "<NullPersistence>"


NULL_PERSISTENCE = NullPersistence()

# 所有存活的 PersistenceManager（弱引用），供 run_leg 从 app.checkpointer 反查后端。
_LIVE_MANAGERS: "weakref.WeakSet[PersistenceManager]" = weakref.WeakSet()


class PersistenceManager(ThreadStoreMixin, AuditStoreMixin):
    """SqliteSaver + thread_meta + audit_event 的统一管理者。

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
        with self._lock, self._conn:
            migrate(self._conn)
        _LIVE_MANAGERS.add(self)

    @property
    def checkpointer(self) -> SqliteSaver:
        return self._saver

    @property
    def db_path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


def persistence_for_checkpointer(checkpointer: Any) -> Optional[PersistenceManager]:
    """Return the live :class:`PersistenceManager` whose saver *is* ``checkpointer``.

    Identity check only: a foreign ``SqliteSaver`` or any other saver returns ``None``.
    Never opens a database.
    """
    if checkpointer is None:
        return None
    for manager in list(_LIVE_MANAGERS):
        if manager._saver is checkpointer:
            return manager
    return None


# ── singleton ─────────────────────────────────────────────────────────

_singleton: Optional[PersistenceManager] = None
_singleton_lock = Lock()


def get_persistence(db_path: Path | None = None) -> PersistenceManager:
    """返回进程内 PersistenceManager 单例；首次调用时按需打开 SQLite。"""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            # ASKANSWER_DB_PATH 可能来自 settings.json 的 db_path：先确保配置已加载
            from ..config.settings import ensure_bootstrapped

            ensure_bootstrapped()
            _singleton = PersistenceManager(db_path or default_db_path())
        return _singleton


def shutdown_persistence() -> None:
    """关闭单例（``atexit`` 注册用）。重复调用 idempotent。"""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.close()
            _singleton = None
