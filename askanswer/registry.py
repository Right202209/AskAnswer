"""Unified tool registry for built-in, MCP, and SQL tools.

Each tool is tagged with one or more ``bundles`` (intent groupings) so the
``answer`` node can bind only the tools relevant to the current intent. Tools
that need human approval (currently the shell tool) are flagged via
``requires_confirmation``; the react subgraph routes those through the
``shell_plan`` HITL flow instead of the regular ``ToolNode`` dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from .mcp import get_manager as _mcp_manager


_LOG = logging.getLogger(__name__)


# Bundle constants — mirrored from intent labels in ``nodes.py``.
BUNDLE_CHAT = "chat"
BUNDLE_SEARCH = "search"
BUNDLE_FILE = "file_read"
BUNDLE_SQL = "sql"
ALL_BUNDLES = frozenset({BUNDLE_CHAT, BUNDLE_SEARCH, BUNDLE_FILE, BUNDLE_SQL})

# Built-in helpers are reachable from every intent so the react loop can
# freely chain file_read / search / chat / sql tools (e.g. SQL mode reading
# a CSV, chat mode searching the web).
_BUILTIN_BUNDLES = ALL_BUNDLES
_SHELL_BUNDLES = frozenset({BUNDLE_CHAT, BUNDLE_SEARCH, BUNDLE_FILE})
# MCP tools come from user-installed servers; expose to all bundles so the
# model can reach them regardless of intent.
_MCP_BUNDLES = ALL_BUNDLES


@dataclass(frozen=True)
class ToolDescriptor:
    tool: BaseTool
    bundles: frozenset[str]
    source: str                       # "builtin" | "shell" | "sql" | "mcp:<server>"
    requires_confirmation: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDescriptor] = {}
        self._lock = Lock()

    def register(self, descriptor: ToolDescriptor) -> None:
        with self._lock:
            self._tools[descriptor.tool.name] = descriptor

    def unregister_source_prefix(self, prefix: str) -> None:
        with self._lock:
            for name in [n for n, d in self._tools.items() if d.source.startswith(prefix)]:
                del self._tools[name]

    def get(self, name: str) -> Optional[ToolDescriptor]:
        with self._lock:
            return self._tools.get(name)

    def list(self, bundle: str | None = None) -> list[BaseTool]:
        with self._lock:
            descriptors = list(self._tools.values())
        if bundle is None:
            return [d.tool for d in descriptors]
        return [d.tool for d in descriptors if bundle in d.bundles]

    def names(self, bundle: str | None = None) -> set[str]:
        return {t.name for t in self.list(bundle)}

    def confirmation_names(self) -> set[str]:
        with self._lock:
            return {n for n, d in self._tools.items() if d.requires_confirmation}

    def refresh_mcp(self) -> None:
        """Rebuild the MCP slice of the registry from the live manager state.

        Call after ``/mcp <url>`` connects or ``/mcp remove <name>``.
        """
        try:
            specs = _mcp_manager().list_tools()
        except Exception as exc:
            _LOG.warning("MCP list_tools failed during registry refresh: %s", exc)
            specs = []

        self.unregister_source_prefix("mcp:")
        for spec in specs:
            wrapped = _wrap_mcp_tool(spec)
            if wrapped is None:
                continue
            self.register(
                ToolDescriptor(
                    tool=wrapped,
                    bundles=_MCP_BUNDLES,
                    source=f"mcp:{spec.get('server', 'unknown')}",
                )
            )


_registry: ToolRegistry | None = None
_registry_lock = Lock()


def get_registry() -> ToolRegistry:
    """Return the process-wide registry, seeding it on first use."""
    global _registry
    with _registry_lock:
        if _registry is None:
            r = ToolRegistry()
            _seed_builtin(r)
            _seed_sql(r)
            r.refresh_mcp()
            _registry = r
        return _registry


# ── Seeding ──────────────────────────────────────────────────────────

def _seed_builtin(registry: ToolRegistry) -> None:
    from .tools import (
        calculate,
        check_weather,
        convert_currency,
        gen_shell_commands_run,
        get_current_time,
        lookup_ip,
        pwd,
        read_file,
        tavily_search,
    )

    plain_tools = (
        check_weather,
        get_current_time,
        calculate,
        convert_currency,
        lookup_ip,
        pwd,
        read_file,
        tavily_search,
    )
    for tool in plain_tools:
        registry.register(
            ToolDescriptor(tool=tool, bundles=_BUILTIN_BUNDLES, source="builtin")
        )

    registry.register(
        ToolDescriptor(
            tool=gen_shell_commands_run,
            bundles=_SHELL_BUNDLES,
            source="shell",
            requires_confirmation=True,
        )
    )


def _seed_sql(registry: ToolRegistry) -> None:
    """Register the natural-language SQL tool. Skips if the module is missing."""
    try:
        from .sqlagent.sql_tool import sql_query
    except ImportError as exc:
        import sys
        print(f"[askanswer] sql_query 工具加载失败: {exc}", file=sys.stderr)
        print("[askanswer] 请检查 langchain>=1.0 是否已安装", file=sys.stderr)
        _LOG.warning("sql_query tool unavailable: %s", exc)
        return
    registry.register(
        ToolDescriptor(
            tool=sql_query,
            bundles=frozenset({BUNDLE_CHAT, BUNDLE_SQL}),
            source="sql",
        )
    )


# ── MCP wrapping ─────────────────────────────────────────────────────

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _jsonschema_to_pydantic(schema: dict, model_name: str) -> type[BaseModel] | None:
    """Best-effort: build a flat pydantic model from a JSON schema.

    Returns ``None`` for non-trivial schemas (anyOf / $ref / nested objects).
    Caller can then fall back to a permissive schema or skip the tool.
    """
    if not isinstance(schema, dict):
        return None
    if schema.get("type") not in (None, "object"):
        return None
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            return None
        if any(k in prop for k in ("anyOf", "allOf", "oneOf", "$ref")):
            return None
        pytype = _JSON_TYPE_MAP.get(prop.get("type", "string"), str)
        description = prop.get("description") or None
        if name in required:
            field_def = (pytype, Field(..., description=description))
        else:
            field_def = (Optional[pytype], Field(default=prop.get("default"), description=description))
        fields[name] = field_def
    if not fields:
        # MCP servers sometimes declare zero-arg tools; pydantic needs at
        # least one field, so add an unused optional placeholder.
        fields["_unused"] = (Optional[str], Field(default=None))
    try:
        return create_model(model_name, **fields)
    except Exception as exc:
        _LOG.debug("create_model failed for %s: %s", model_name, exc)
        return None


def _wrap_mcp_tool(spec: dict) -> BaseTool | None:
    name = spec.get("name")
    if not name:
        return None
    description = spec.get("description") or name
    args_schema = _jsonschema_to_pydantic(spec.get("input_schema") or {}, f"{name}_args")
    if args_schema is None:
        _LOG.warning("MCP tool %s has unsupported schema; skipping", name)
        return None

    def _proxy(**kwargs: Any) -> str:
        kwargs.pop("_unused", None)
        try:
            return _mcp_manager().call_tool(name, kwargs)
        except Exception as exc:
            return f"MCP 工具 {name} 调用失败：{exc}"

    try:
        return StructuredTool.from_function(
            func=_proxy,
            name=name,
            description=description,
            args_schema=args_schema,
        )
    except Exception as exc:
        _LOG.warning("Failed to wrap MCP tool %s: %s", name, exc)
        return None
