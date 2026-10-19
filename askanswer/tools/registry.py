"""Unified tool registry: the single source of truth for built-in, subagent and MCP tools.

Every tool carries ``tags`` (intent names double as tags); the ``answer`` node binds
the tools matching the active intent handler's ``bundle_tags``.  Tools that need
human approval declare a ``confirmation_class`` (``shell`` / ``fs_write`` /
``external_api_paid``) and are routed through the HITL protocol in
:mod:`askanswer.core.confirmations` instead of a plain ``ToolNode``.

Extending: ``get_registry().register(ToolDescriptor(tool=..., tags=frozenset({...}),
source="myapp"))``.

统一工具注册表：内置工具、MCP 工具、SQL 工具的唯一真相源。

每个工具会贴上一个或多个 ``tags`` 标签（intent 名也是 tag），``answer`` 节点
按当前 handler 的 tag 集合取工具绑定给 LLM。需要人工确认的工具通过
``confirmation_class`` 标记（shell / fs_write / external_api_paid 三类均已接入
执行体，见 ``confirmations.py``），react 子图会把它们路由到 ``confirm_plan``
HITL 流程，而非走普通 ``ToolNode`` 直接执行。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Optional

from langchain_core.tools import BaseTool

from .mcp_adapter import wrap_mcp_tool
from .mcp_manager import get_manager as _mcp_manager

_LOG = logging.getLogger(__name__)


# Intent tag 常量（intent 名同时也是工具 tag）。
TAG_CHAT = "chat"
TAG_SEARCH = "search"
TAG_FILE = "file_read"
TAG_SQL = "sql"
TAG_MATH = "math"
TAG_HELIX = "helix"
TAG_RESEARCH = "research"
TAG_DECISION = "decision"
ALL_INTENT_TAGS = frozenset(
    {TAG_CHAT, TAG_SEARCH, TAG_FILE, TAG_SQL, TAG_MATH, TAG_HELIX, TAG_RESEARCH, TAG_DECISION}
)

# 内置工具默认在所有 intent tag 中可用，方便 react 循环跨 intent 串工具
# （例如 SQL 模式下也能读 CSV、chat 模式下也能联网搜索）。
_BUILTIN_TAGS = ALL_INTENT_TAGS | frozenset({"builtin", "io_bound"})
# Shell 工具刻意从 sql tag 中剔除，让 SQL 流程更聚焦、避免误调用 shell。
_SHELL_TAGS = frozenset({TAG_CHAT, TAG_SEARCH, TAG_FILE, "shell"})
# MCP 工具来自用户安装的外部服务。默认仅暴露给 chat intent，避免在 sql/file_read/
# math/search 等更具体的流程被 prompt 注入诱导调用付费/写副作用 API。
# 用户如确认要在所有 intent 下使用 MCP，可设环境变量 ASKANSWER_MCP_ALL_INTENTS=1。
# 该开关在每次 ``refresh_mcp()`` 时读取（而非导入期固化），运行中修改 env 后再 refresh 即生效。
_MCP_ALL_INTENTS_ENV = "ASKANSWER_MCP_ALL_INTENTS"
_MCP_TAGS_ALL = ALL_INTENT_TAGS | frozenset({"mcp", "external_api"})
_MCP_TAGS_CHAT_ONLY = frozenset({TAG_CHAT, "mcp", "external_api"})


def mcp_tags() -> frozenset[str]:
    """MCP 工具当前应挂的 tag 集合（按 ``ASKANSWER_MCP_ALL_INTENTS`` 动态决定）。"""
    raw = (os.environ.get(_MCP_ALL_INTENTS_ENV) or "").strip().lower()
    return _MCP_TAGS_ALL if raw in {"1", "true", "yes"} else _MCP_TAGS_CHAT_ONLY


ConfirmationClass = Literal["none", "shell", "fs_write", "external_api_paid"]


@dataclass(frozen=True)
class ToolDescriptor:
    """A registered tool: the LangChain tool, its tags, source and confirmation class."""

    # 真正的工具对象（langchain BaseTool）
    tool: BaseTool
    # 该工具暴露给哪些 tag；intent 名也作为 tag 使用
    tags: frozenset[str]
    # 来源标签，便于按前缀批量摘除（如 "mcp:" 重连时清掉旧的）
    source: str                       # "builtin" | "shell" | "sql" | "mcp:<server>"
    # 需要哪类人工确认；三类的规划/闸门/执行体都在 confirmations.py
    confirmation_class: ConfirmationClass = "none"


class ToolRegistry:
    """Thread-safe tool registry (线程安全的工具注册表)."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDescriptor] = {}
        self._lock = Lock()

    def register(self, descriptor: ToolDescriptor) -> None:
        """按工具名注册或覆盖一个工具描述符。"""
        with self._lock:
            self._tools[descriptor.tool.name] = descriptor

    def unregister_source_prefix(self, prefix: str) -> None:
        """按 source 前缀批量摘除（用于重新拉取 MCP 工具列表前先清理旧条目）。"""
        with self._lock:
            for name in [n for n, d in self._tools.items() if d.source.startswith(prefix)]:
                del self._tools[name]

    def get(self, name: str) -> Optional[ToolDescriptor]:
        with self._lock:
            return self._tools.get(name)

    def list(
        self,
        *,
        tags: set[str] | frozenset[str] | None = None,
    ) -> list[BaseTool]:
        """列出所有工具；指定 ``tags`` 时只返回命中任一 tag 的工具。"""
        with self._lock:
            descriptors = list(self._tools.values())
        if tags is None:
            return [d.tool for d in descriptors]
        wanted = frozenset(tags)
        return [d.tool for d in descriptors if d.tags & wanted]

    def names(self, *, tags: set[str] | frozenset[str] | None = None) -> set[str]:
        return {t.name for t in self.list(tags=tags)}

    def confirmation_names(self) -> set[str]:
        """返回所有标记为“需要确认”的工具名集合，供 react 子图路由判断。"""
        with self._lock:
            return {n for n, d in self._tools.items() if d.confirmation_class != "none"}

    def confirmation_classes(self) -> dict[str, str]:
        """返回 tool name -> confirmation class（react 子图据此分发 HITL 流程）。"""
        with self._lock:
            return {
                n: d.confirmation_class
                for n, d in self._tools.items()
                if d.confirmation_class != "none"
            }

    def refresh_mcp(self) -> None:
        """从 MCP 管理器实时同步工具，重建注册表中 mcp 那一片。

        宿主连接、断开或重新探测 MCP 服务后调用。
        健康探测标记为 ``disconnected`` 的 server，其工具会被跳过 —— 避免把已失联
        server 的死工具继续暴露给 LLM；server 恢复后再次 refresh 即可重新出现。
        """
        try:
            specs = _mcp_manager().list_tools()
        except Exception as exc:
            # 拉取失败不阻塞主流程，记录 warning 后按空列表处理
            _LOG.warning("MCP list_tools failed during registry refresh: %s", exc)
            specs = []

        # 收集处于 disconnected 状态的 server 名，稍后据此过滤其工具。
        try:
            offline = {
                s["name"]
                for s in _mcp_manager().list_servers()
                if s.get("status") == "disconnected"
            }
        except Exception:
            offline = set()

        # 先把旧的 mcp:* 条目摘掉，避免出现已断开服务的残留工具
        self.unregister_source_prefix("mcp:")
        tags = mcp_tags()
        for spec in specs:
            if spec.get("server") in offline:
                continue
            wrapped = wrap_mcp_tool(spec)
            if wrapped is None:
                continue
            self.register(
                ToolDescriptor(
                    tool=wrapped,
                    tags=tags,
                    source=f"mcp:{spec.get('server', 'unknown')}",
                )
            )


# 进程级单例 —— 注册表是全局的，所有节点共享同一份。
_registry: ToolRegistry | None = None
_registry_lock = Lock()


def get_registry() -> ToolRegistry:
    """Return the process-wide registry, seeding built-in, subagent and MCP tools on first use.

    (获取进程级注册表；首次调用时按需 seed 内置 + 子 agent + MCP 工具。)
    """
    global _registry
    with _registry_lock:
        if _registry is None:
            r = ToolRegistry()
            _seed_builtin(r)
            _seed_sql(r)
            _seed_helix(r)
            _seed_research(r)
            _seed_decision(r)
            r.refresh_mcp()
            _registry = r
        return _registry


# ── Seeding ──────────────────────────────────────────────────────────

def _seed_builtin(registry: ToolRegistry) -> None:
    """注册所有内置工具。延迟 import：工具模块依赖模型代理，避免导入期循环。"""
    from .builtin import (
        calculate,
        check_weather,
        convert_currency,
        get_current_time,
        lookup_ip,
        pwd,
        tavily_search,
    )
    from .filesystem import read_file, write_file
    from .shell import gen_shell_commands_run

    # 普通工具：所有 bundle 都可见
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
            ToolDescriptor(tool=tool, tags=_BUILTIN_TAGS, source="builtin")
        )

    # Shell 工具特殊：bundle 不含 sql + 需要人工确认
    registry.register(
        ToolDescriptor(
            tool=gen_shell_commands_run,
            tags=_SHELL_TAGS,
            source="shell",
            confirmation_class="shell",
        )
    )

    # 写文件工具：仅暴露给 chat / file_read，落盘前必须经过 fs_write 人机确认
    # （diff 预览 + 敏感路径/大小闸门），不给 search/sql/math/helix 流程误用的机会。
    registry.register(
        ToolDescriptor(
            tool=write_file,
            tags=frozenset({TAG_CHAT, TAG_FILE, "fs_write"}),
            source="builtin",
            confirmation_class="fs_write",
        )
    )


def _seed_sql(registry: ToolRegistry) -> None:
    """注册自然语言 SQL 工具；模块缺失（例如未装 langchain>=1.0）时跳过。"""
    try:
        from ..subagents.sql.sql_tool import sql_query
    except ImportError as exc:
        # 不抛异常：让其它工具仍然可用，并通过 stderr 给出明确提示
        import sys
        print(f"[askanswer] sql_query 工具加载失败: {exc}", file=sys.stderr)
        print("[askanswer] 请检查 langchain>=1.0 是否已安装", file=sys.stderr)
        _LOG.warning("sql_query tool unavailable: %s", exc)
        return
    # SQL 工具仅暴露给 chat 与 sql 两个 bundle，避免在 search/file_read 中干扰判断
    registry.register(
        ToolDescriptor(
            tool=sql_query,
            tags=frozenset({TAG_CHAT, TAG_SQL, "sql_tool"}),
            source="sql",
        )
    )


def _seed_helix(registry: ToolRegistry) -> None:
    """注册 Helix 规格演化工具；模块缺失时跳过，不影响其他工具。"""
    try:
        from ..subagents.helix.helix_tool import helix_spec_loop
    except ImportError as exc:
        import sys
        print(f"[askanswer] helix_spec_loop 工具加载失败: {exc}", file=sys.stderr)
        _LOG.warning("helix_spec_loop tool unavailable: %s", exc)
        return
    # 仅暴露给 chat / helix bundle：避免 search / sql / file_read 等流程下被误调用。
    registry.register(
        ToolDescriptor(
            tool=helix_spec_loop,
            tags=frozenset({TAG_CHAT, TAG_HELIX, "helix_tool"}),
            source="helix",
        )
    )


def _seed_research(registry: ToolRegistry) -> None:
    """注册多源研究简报工具；模块缺失时跳过，不影响其他工具。"""
    try:
        from ..subagents.research.research_tool import research_brief_loop
    except ImportError as exc:
        import sys
        print(f"[askanswer] research_brief_loop 工具加载失败: {exc}", file=sys.stderr)
        _LOG.warning("research_brief_loop tool unavailable: %s", exc)
        return
    # 仅暴露给 chat / research bundle：避免 sql / file_read / math 等流程下被误调用。
    registry.register(
        ToolDescriptor(
            tool=research_brief_loop,
            tags=frozenset({TAG_CHAT, TAG_RESEARCH, "research_tool"}),
            source="research",
        )
    )


def _seed_decision(registry: ToolRegistry) -> None:
    """注册决策备忘工具；模块缺失时跳过，不影响其他工具。"""
    try:
        from ..subagents.decision.decision_tool import decision_memo_loop
    except ImportError as exc:
        import sys
        print(f"[askanswer] decision_memo_loop 工具加载失败: {exc}", file=sys.stderr)
        _LOG.warning("decision_memo_loop tool unavailable: %s", exc)
        return
    # 仅暴露给 chat / decision bundle：避免其它更精准的流程下被误调用。
    registry.register(
        ToolDescriptor(
            tool=decision_memo_loop,
            tags=frozenset({TAG_CHAT, TAG_DECISION, "decision_tool"}),
            source="decision",
        )
    )
