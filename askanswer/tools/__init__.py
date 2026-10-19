"""Tool layer: the registry plus built-in, filesystem, shell and MCP tools.

- :mod:`.registry` – :class:`ToolRegistry`, :class:`ToolDescriptor`, :func:`get_registry`
  (seeds built-ins, subagent tools and MCP tools on first use);
- :mod:`.builtin` – read-only utility tools (weather, time, calculator, currency, IP,
  cwd, Tavily search);
- :mod:`.filesystem` – ``read_file`` / ``write_file`` and their path / size gates;
- :mod:`.shell` – ``gen_shell_commands_run`` and the dangerous-command gate;
- :mod:`.mcp_manager` / :mod:`.mcp_adapter` – MCP client sessions and tool wrapping.
"""

from .registry import (
    ALL_INTENT_TAGS,
    ConfirmationClass,
    ToolDescriptor,
    ToolRegistry,
    get_registry,
)

__all__ = [
    "ALL_INTENT_TAGS",
    "ConfirmationClass",
    "ToolDescriptor",
    "ToolRegistry",
    "get_registry",
]
