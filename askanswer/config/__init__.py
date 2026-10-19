"""Configuration: layered ``settings.json`` / ``.env`` loading and the runtime context.

- :mod:`.settings` – settings discovery, merge and expansion into env vars
  (``bootstrap_environ``), plus helpers such as :func:`~.settings.resolve_db_dsn`;
- :mod:`.schema` – :class:`~.schema.ContextSchema`, the per-invocation runtime
  context passed to every node and tool.
"""

from .schema import ContextSchema, normalize_context

__all__ = ["ContextSchema", "normalize_context"]
