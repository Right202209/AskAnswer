"""LangGraph 父图共享的状态定义（架构文档 P4 / D3）。

全局状态只保留与技能无关的通用字段；每个 intent 自己的参数（搜索词、文件路径……）
统一放进 ``slots``，新增 intent 不需要改这里。``step`` 只能取 :data:`Phase` 中的值。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

# 流程阶段：取代过去的自由字符串
Phase = Literal["understood", "tool_called", "completed", "retry_search"]
PHASE_UNDERSTOOD: Phase = "understood"      # understand 刚完成分类（本轮首次进入 answer）
PHASE_TOOL_CALLED: Phase = "tool_called"    # answer 产出 tool_calls，等待工具执行
PHASE_COMPLETED: Phase = "completed"        # 已产出最终答案
PHASE_RETRY: Phase = "retry_search"         # sorcery 判定需要重试
PHASES: frozenset[str] = frozenset(
    {PHASE_UNDERSTOOD, PHASE_TOOL_CALLED, PHASE_COMPLETED, PHASE_RETRY}
)

# 常用 slot 名（只是约定，不是全集；intent 可以写任意自己的 slot）
SLOT_SEARCH_QUERY = "search_query"
SLOT_FILE_PATH = "file_path"

# 允许 intent 的澄清选项 / 重试指令直接改写的通用字段；其余键一律视为 slot
GENERIC_UPDATE_FIELDS = frozenset({"intent", "user_query"})


class SearchState(TypedDict, total=False):
    # 对话历史：add_messages reducer 追加而非覆盖。只含用户、模型、工具消息，
    # 不含进度提示（P6）。
    messages: Annotated[list, add_messages]
    user_query: str          # 对用户需求的复述（understand 写入，澄清可细化）
    intent: str              # 本轮 intent：file_read | search | chat | sql | math | 插件 intent
    slots: dict              # intent 专用参数，如 {"search_query": ..., "file_path": ...}
    final_answer: str        # 最终给用户的答案
    retry_count: int         # 已触发的重试次数
    step: Phase              # 当前阶段
    retry_directive: dict    # sorcery 给下一轮 answer 的结构化重试指令，answer 消费后清空
    pending_confirmations: dict  # {tool_call_id: {"class": ..., 规划载荷, decision|blocked}}


def get_slot(state: Mapping[str, Any], name: str, default: str = "") -> Any:
    """读取一个 slot；state 或 slots 缺失时返回默认值。"""
    slots = state.get("slots") or {}
    value = slots.get(name, default)
    return default if value is None else value


def with_slots(state: Mapping[str, Any], **updates: Any) -> dict:
    """返回合并后的新 slots 字典（节点用它构造部分状态 ``{"slots": ...}``）。"""
    merged = dict(state.get("slots") or {})
    merged.update(updates)
    return merged


def apply_updates(state: Mapping[str, Any], updates: Mapping[str, Any]) -> dict:
    """把 intent 给出的 updates 拆成部分状态：通用字段直写，其余写进 slots。"""
    out: dict = {}
    slot_updates: dict = {}
    for key, value in updates.items():
        if key in GENERIC_UPDATE_FIELDS:
            out[key] = value
        else:
            slot_updates[key] = value
    if slot_updates:
        out["slots"] = with_slots(state, **slot_updates)
    return out
