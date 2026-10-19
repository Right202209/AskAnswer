"""Persistence layer: SQLite checkpoints, thread metadata and audit events.

Public entry points: :func:`get_persistence`, :func:`shutdown_persistence`,
:class:`PersistenceManager`, plus the row views :class:`ThreadMeta` and
:class:`AuditEvent`.  Nothing here opens a database at import time.
"""

from .audit_store import AuditEvent
from .manager import (
    NULL_PERSISTENCE,
    NullPersistence,
    PersistenceBackend,
    PersistenceManager,
    default_db_path,
    get_persistence,
    persistence_for_checkpointer,
    shutdown_persistence,
)
from .migrations import SCHEMA_VERSION
from .threads import ThreadMeta

__all__ = [
    "NULL_PERSISTENCE",
    "SCHEMA_VERSION",
    "AuditEvent",
    "NullPersistence",
    "PersistenceBackend",
    "PersistenceManager",
    "ThreadMeta",
    "default_db_path",
    "get_persistence",
    "persistence_for_checkpointer",
    "shutdown_persistence",
]
