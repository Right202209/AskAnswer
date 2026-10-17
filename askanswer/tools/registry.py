"""统一工具注册表：内置工具、MCP 工具、SQL 工具的唯一真相源。

每个工具会贴上一个或多个 ``tags`` 标签（intent 名也是 tag），``answer`` 节点
按当前 handler 的 tag 集合取工具绑定给 LLM。需要人工确认的工具通过
``confirmation_class`` 标记（shell / fs_write / external_api_paid 三类均已接入
执行体，见 ``confirmations.py``），主图会把它们路由到 ``plan_approvals``
HITL 流程，而非走普通 ``ToolNode`` 直接执行。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from askanswer.config import current_settings
from askanswer.core.types import (  # noqa: F401 - 对外再导出
    CAP_AGENT_DECISION,
    CAP_AGENT_RESEARCH,
    CAP_AGENT_SPEC,
    CAP_DATA_SQL,
    CAP_EXEC_SHELL,
    CAP_FS_READ,
    CAP_FS_WRITE,
    CAP_MCP,
    CAP_NET_LOOKUP,
    CAP_UTIL,
    CAP_WEB_SEARCH,
    RISK_EXEC,
    RISK_EXTERNAL,
    RISK_NETWORK,
    RISK_PAID,
    RISK_READ,
    RISK_WRITE,
    RISKS_REQUIRING_APPROVAL,
    Risk,
)
from askanswer.mcp.manager import get_manager as _mcp_manager

_LOG = logging.getLogger(__name__)


# Intent tag 常量。保留 BUNDLE_* 别名，兼容旧调用方。
TAG_CHAT = BUNDLE_CHAT = "chat"
TAG_SEARCH = BUNDLE_SEARCH = "search"
TAG_FILE = BUNDLE_FILE = "file_read"
TAG_SQL = BUNDLE_SQL = "sql"
TAG_MATH = "math"
TAG_HELIX = "helix"
TAG_RESEARCH = "research"
TAG_DECISION = "decision"
ALL_INTENT_TAGS = frozenset(
    {TAG_CHAT, TAG_SEARCH, TAG_FILE, TAG_SQL, TAG_MATH, TAG_HELIX, TAG_RESEARCH, TAG_DECISION}
)
ALL_BUNDLES = ALL_INTENT_TAGS

# 内置工具默认在所有 intent tag 中可用，方便 react 循环跨 intent 串工具
# （例如 SQL 模式下也能读 CSV、chat 模式下也能联网搜索）。
_BUILTIN_TAGS = ALL_INTENT_TAGS | frozenset({"builtin", "io_bound"})
# Shell 工具刻意从 sql tag 中剔除，让 SQL 流程更聚焦、避免误调用 shell。
_SHELL_TAGS = frozenset({TAG_CHAT, TAG_SEARCH, TAG_FILE, "shell"})
# MCP 工具来自用户安装的外部服务。默认仅暴露给 chat intent，避免在 sql/file_read/
# math/search 等更具体的流程被 prompt 注入诱导调用付费/写副作用 API。
# 用户如确认要在所有 intent 下使用 MCP，可设环境变量
# ASKANSWER_MCP_ALL_INTENTS=1 重新放开（兼容旧行为）。
# 运行时读取（而非导入时固化），保证 settings.json 注入的值也能生效（P1/P3）。
def _mcp_tags() -> frozenset[str]:
    if current_settings().mcp_all_intents:
        return ALL_INTENT_TAGS | frozenset({"mcp", "external_api"})
    return frozenset({TAG_CHAT, "mcp", "external_api"})

ConfirmationClass = Literal["none", "shell", "fs_write", "external_api_paid"]

# ── 风险等级与能力（架构文档 P13 / §7） ─────────────────────────────────
# tags 回答「哪个 intent 能看到这个工具」；risk/capability 回答「这个工具能做什么、
# 有多危险」。权限语义不再靠工具名或技能名的字符串相等来推断。

# 用户可通过 ASKANSWER_DENY_TOOLS（或 settings.json 的 env 块）禁用任意工具，逗号分隔。


class ToolRegistrationError(ValueError):
    """工具描述符违反注册不变量（如高风险工具未声明确认类）。"""


def denied_tool_names() -> frozenset[str]:
    return current_settings().deny_tools


@dataclass(frozen=True)
class ToolDescriptor:
    # 真正的工具对象（langchain BaseTool）
    tool: BaseTool
    # 该工具暴露给哪些 tag；intent 名也作为 tag 使用
    tags: frozenset[str]
    # 来源标签，便于按前缀批量摘除（如 "mcp:" 重连时清掉旧的）
    source: str                       # "builtin" | "shell" | "sql" | "mcp:<server>"
    # 需要哪类人工确认；三类的规划/闸门/执行体都在 confirmations.py
    confirmation_class: ConfirmationClass = "none"
    # 风险等级：write / exec / paid 必须配确认类
    risk: Risk = RISK_READ
    # 能力标签，如 fs.read / web.search；用于权限展示与按能力过滤
    capability: str = CAP_UTIL

    def __post_init__(self) -> None:
        if self.risk in RISKS_REQUIRING_APPROVAL and self.confirmation_class == "none":
            name = getattr(self.tool, "name", "?")
            raise ToolRegistrationError(
                f"工具 {name} 的风险等级为 {self.risk}，必须声明 confirmation_class"
            )

    @property
    def bundles(self) -> frozenset[str]:
        """Compatibility alias for older code that still says bundle."""
        return self.tags

    @property
    def requires_confirmation(self) -> bool:
        """Compatibility alias for older callers."""
        return self.confirmation_class != "none"


class ToolRegistry:
    """线程安全的工具注册表（单进程内的工具集合的真相源）。"""

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
        bundle: str | None = None,
        *,
        tags: set[str] | frozenset[str] | None = None,
        capabilities: set[str] | frozenset[str] | None = None,
    ) -> list[BaseTool]:
        """列出工具；指定 tags/bundle 时只返回命中任一 tag 的工具。

        ``ASKANSWER_DENY_TOOLS`` 里列出的工具一律不返回（用户可关闭任何工具）。
        """
        return [d.tool for d in self.descriptors(bundle, tags=tags, capabilities=capabilities)]

    def descriptors(
        self,
        bundle: str | None = None,
        *,
        tags: set[str] | frozenset[str] | None = None,
        capabilities: set[str] | frozenset[str] | None = None,
    ) -> list[ToolDescriptor]:
        with self._lock:
            descriptors = list(self._tools.values())
        denied = denied_tool_names()
        if denied:
            descriptors = [d for d in descriptors if d.tool.name not in denied]
        if tags is None and bundle is not None:
            tags = frozenset({bundle})
        if tags is not None:
            wanted = frozenset(tags)
            descriptors = [d for d in descriptors if d.tags & wanted]
        if capabilities is not None:
            caps = frozenset(capabilities)
            descriptors = [d for d in descriptors if d.capability in caps]
        return descriptors

    def names(self, bundle: str | None = None) -> set[str]:
        return {t.name for t in self.list(bundle)}

    def confirmation_names(self) -> set[str]:
        """返回所有标记为“需要确认”的工具名集合，供主图路由判断。"""
        with self._lock:
            return {n for n, d in self._tools.items() if d.confirmation_class != "none"}

    def confirmation_classes(self) -> dict[str, str]:
        """返回 tool name -> confirmation class（主图据此分发审批流程）。"""
        with self._lock:
            return {
                n: d.confirmation_class
                for n, d in self._tools.items()
                if d.confirmation_class != "none"
            }

    def refresh_mcp(self) -> None:
        """从 MCP 管理器实时同步工具，重建注册表中 mcp 那一片。

        在 ``/mcp <url>`` 连接成功或 ``/mcp remove <name>`` 之后调用。
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
        for spec in specs:
            if spec.get("server") in offline:
                continue
            wrapped = _wrap_mcp_tool(spec)
            if wrapped is None:
                continue
            self.register(
                ToolDescriptor(
                    tool=wrapped,
                    tags=_mcp_tags(),
                    source=f"mcp:{spec.get('server', 'unknown')}",
                    risk=RISK_EXTERNAL,
                    capability=CAP_MCP,
                )
            )


# 进程级单例 —— 注册表是全局的，所有节点共享同一份。
_registry: ToolRegistry | None = None
_registry_lock = Lock()


def get_registry() -> ToolRegistry:
    """获取进程级注册表；首次调用时按需 seed 内置 + SQL + Helix + MCP 工具。"""
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
    """注册所有内置工具。延迟 import 以避免循环依赖（tools.py 也会反向用到 model）。"""
    from askanswer.tools.builtin import (
        calculate,
        check_weather,
        convert_currency,
        gen_shell_commands_run,
        get_current_time,
        lookup_ip,
        pwd,
        read_file,
        tavily_search,
        write_file,
    )

    # 普通工具：所有 bundle 都可见；风险等级与能力逐个声明
    plain_tools = (
        (check_weather, RISK_NETWORK, CAP_NET_LOOKUP),
        (get_current_time, RISK_READ, CAP_UTIL),
        (calculate, RISK_READ, CAP_UTIL),
        (convert_currency, RISK_NETWORK, CAP_NET_LOOKUP),
        (lookup_ip, RISK_NETWORK, CAP_NET_LOOKUP),
        (pwd, RISK_READ, CAP_UTIL),
        (read_file, RISK_READ, CAP_FS_READ),
        (tavily_search, RISK_NETWORK, CAP_WEB_SEARCH),
    )
    for tool, risk, capability in plain_tools:
        registry.register(
            ToolDescriptor(
                tool=tool, tags=_BUILTIN_TAGS, source="builtin",
                risk=risk, capability=capability,
            )
        )

    # Shell 工具特殊：bundle 不含 sql + 需要人工确认
    registry.register(
        ToolDescriptor(
            tool=gen_shell_commands_run,
            tags=_SHELL_TAGS,
            source="shell",
            confirmation_class="shell",
            risk=RISK_EXEC,
            capability=CAP_EXEC_SHELL,
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
            risk=RISK_WRITE,
            capability=CAP_FS_WRITE,
        )
    )


def _seed_sql(registry: ToolRegistry) -> None:
    """注册自然语言 SQL 工具；模块缺失（例如未装 langchain>=1.0）时跳过。"""
    try:
        from askanswer.agents.sql.sql_tool import sql_query
    except ImportError as exc:
        # 不抛异常：让其它工具仍然可用，仅在终端给出明确提示
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
            risk=RISK_READ,
            capability=CAP_DATA_SQL,
        )
    )


def _seed_helix(registry: ToolRegistry) -> None:
    """注册 Helix 规格演化工具；模块缺失时跳过，不影响其他工具。"""
    try:
        from askanswer.agents.helix.helix_tool import helix_spec_loop
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
            risk=RISK_READ,
            capability=CAP_AGENT_SPEC,
        )
    )


def _seed_research(registry: ToolRegistry) -> None:
    """注册多源研究简报工具；模块缺失时跳过，不影响其他工具。"""
    try:
        from askanswer.agents.research.research_tool import research_brief_loop
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
            risk=RISK_NETWORK,
            capability=CAP_AGENT_RESEARCH,
        )
    )


def _seed_decision(registry: ToolRegistry) -> None:
    """注册决策备忘工具；模块缺失时跳过，不影响其他工具。"""
    try:
        from askanswer.agents.decision.decision_tool import decision_memo_loop
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
            risk=RISK_NETWORK,
            capability=CAP_AGENT_DECISION,
        )
    )


# ── MCP wrapping ─────────────────────────────────────────────────────

# JSON Schema 类型 → Python 类型的简单映射
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _jsonschema_to_pydantic(schema: dict, model_name: str) -> type[BaseModel] | None:
    """尽力把 MCP 工具的 input_schema 转换成扁平的 pydantic 模型。

    遇到 ``anyOf`` / ``$ref`` / 嵌套对象等复杂结构时返回 ``None`` —— 调用方据此
    回退到宽松 schema 或干脆跳过该工具，避免因 schema 不兼容导致整体崩溃。
    """
    if not isinstance(schema, dict):
        return None
    # 只支持顶层为 object（或未声明 type）的简单 schema
    if schema.get("type") not in (None, "object"):
        return None
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            return None
        # 复杂关键字直接放弃，避免转换出错或字段语义不准
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
        # 一些 MCP server 会暴露零参工具；pydantic 至少要一个字段，加个占位
        fields["_unused"] = (Optional[str], Field(default=None))
    try:
        return create_model(model_name, **fields)
    except Exception as exc:
        _LOG.debug("create_model failed for %s: %s", model_name, exc)
        return None


def _wrap_mcp_tool(spec: dict) -> BaseTool | None:
    """把 MCP server 描述的一个工具包装成 langchain ``StructuredTool``。"""
    name = spec.get("name")
    if not name:
        return None
    description = spec.get("description") or name
    # 自动从 JSON Schema 派生 pydantic 入参模型；不支持时跳过此工具
    args_schema = _jsonschema_to_pydantic(spec.get("input_schema") or {}, f"{name}_args")
    if args_schema is None:
        _LOG.warning("MCP tool %s has unsupported schema; skipping", name)
        return None

    def _proxy(**kwargs: Any) -> str:
        # 占位字段不能传给 MCP server
        kwargs.pop("_unused", None)
        try:
            return _mcp_manager().call_tool(name, kwargs)
        except Exception as exc:
            # 工具调用失败包成普通字符串返回，让 LLM 看到错误内容并自行决策
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
