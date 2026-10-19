"""MCP tool adapter: JSON-Schema → Pydantic args model → LangChain ``StructuredTool``.

Each wrapped tool proxies calls to :func:`askanswer.tools.mcp_manager.get_manager`.
Tools whose schema cannot be represented as a flat model are skipped (logged).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from .mcp_manager import get_manager as _mcp_manager

_LOG = logging.getLogger(__name__)


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


def jsonschema_to_pydantic(schema: dict, model_name: str) -> type[BaseModel] | None:
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
    # 零参工具（properties 为空）得到一个无字段模型，Pydantic v2 / StructuredTool 都支持。
    # 旧版用 "_unused" 占位字段，但 Pydantic v2 拒绝下划线字段名，反而让零参工具被静默跳过。
    try:
        return create_model(model_name, **fields)
    except Exception as exc:
        _LOG.debug("create_model failed for %s: %s", model_name, exc)
        return None


def wrap_mcp_tool(spec: dict) -> BaseTool | None:
    """把 MCP server 描述的一个工具包装成 langchain ``StructuredTool``。"""
    name = spec.get("name")
    if not name:
        return None
    description = spec.get("description") or name
    # 自动从 JSON Schema 派生 pydantic 入参模型；不支持时跳过此工具
    args_schema = jsonschema_to_pydantic(spec.get("input_schema") or {}, f"{name}_args")
    if args_schema is None:
        _LOG.warning("MCP tool %s has unsupported schema; skipping", name)
        return None

    def _proxy(**kwargs: Any) -> str:
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
