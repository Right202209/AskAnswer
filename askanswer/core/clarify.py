"""answer 之前的通用澄清节点（generic clarification）。

与 ``confirmations`` 同构的一段 HITL：intent handler 通过可选的 ``clarify()`` 声明
「回答前需要先向用户澄清什么」（缺文件路径、缺数据库 DSN、研究范围不清等），本节点
在 react 子图入口把它物化成 ``interrupt()``，由宿主应用收集选择，再把结果并回
SearchState，然后才进入 answer。

放在独立模块（而非塞进 ``core/react/dispatch.py``）的原因：让 react 内部聚焦
answer⇄tools 主循环，也避免该文件继续膨胀。

不变量：
- **父图拓扑不变**——澄清节点只加在 answer 子图内部，父图仍是
  understand → answer → sorcery。
- **只在首轮触发**——用 ``step == "understood"`` 把 sorcery 重试
  （step=retry_search）与 react 内部 answer⇄tools 循环挡在门外，确保每轮问答最多
  澄清一次。
- **无需重跑 LLM**——intent 早在 understand 阶段就落进 state，本节点仅读已提交的
  state 与运行时 context，interrupt 恢复后重算是廉价且确定的。
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from ..config.schema import ContextSchema, normalize_context
from ..intents import get_intent_registry
from ..intents.base import ClarificationRequest, get_clarification
from .state import SearchState

# understand 节点把 step 置为该值；澄清只在「首次进入 answer」这一轮触发。
_FIRST_PASS_STEP = "understood"
# interrupt 载荷的 type，宿主应用据此选择对应的交互方式。
_INTERRUPT_TYPE = "clarify"


def clarify_node(state: SearchState, runtime: Runtime[ContextSchema]) -> dict:
    """react 子图入口：按当前 intent handler 的 ``clarify()`` 决定是否发起澄清。"""
    if state.get("step") != _FIRST_PASS_STEP:
        return {}  # 重试轮 / 非首轮：不澄清
    handler = get_intent_registry().get(state.get("intent", "search"))
    context = normalize_context(getattr(runtime, "context", None))
    request = get_clarification(handler, state, context)
    if request is None:
        return {}  # 该 intent 无需澄清（或未实现 clarify）
    decision = interrupt(_interrupt_payload(request))
    return _resolve(request, decision)


def _interrupt_payload(request: ClarificationRequest) -> dict:
    """把 ClarificationRequest 压成宿主应用可消费的纯 dict。"""
    return {
        "type": _INTERRUPT_TYPE,
        "prompt": request.prompt,
        "labels": [choice.label for choice in request.choices],
        "default_index": request.default_index,
        "free_text": bool(request.free_text_field),
        "free_text_label": request.free_text_label,
        "free_text_prompt": request.free_text_prompt,
    }


def _resolve(request: ClarificationRequest, decision: Any) -> dict:
    """把宿主应用传回的 ``{"index","text"}`` 映射成状态更新。"""
    index, text = _read_decision(decision)
    free_slot = len(request.choices)  # 手动输入项恒排在所有 choice 之后
    if request.free_text_field and index == free_slot:
        return {request.free_text_field: text} if text else {}
    if 0 <= index < len(request.choices):
        return dict(request.choices[index].updates)
    return {}  # CANCELLED / 越界：保持现状


def _read_decision(decision: Any) -> tuple[int, str]:
    """解析 resume 形态，取出 ``(index, free_text)``。"""
    if isinstance(decision, dict):
        try:
            index = int(decision.get("index", -1))
        except (TypeError, ValueError):
            index = -1
        text = str(decision.get("text") or "").strip()
        return index, text
    return -1, ""
