# Graph 运行器：UI 无关的"一腿"事件流（C1 计划中的 runner 先行落地，C3 复用）。
#
# 事件契约（与 cli.stream_query 的消费口径同源，front-end 只做渲染/编码）：
# - token     LLM 面向用户的增量文本；
# - tool      LLM 开始规划工具调用（消费者应清空 token 缓冲——规划期文本不属于最终答案，
#             与 CLI _handle_message_chunk 的 in_tool 语义等价）；
# - node      图节点完成（data=原始 update dict，可能含消息对象等非 JSON 值，
#             由传输层负责裁剪，见 wire.py）；
# - interrupt 图挂起等待人工输入（data=interrupt 载荷，经 resume_input(...) 续跑）；
# - final     一腿正常结束（text=最终答案，含 state 兜底）。
#
# 一"腿"（leg）= 从一次图输入（新问题或 resume 决定）跑到 interrupt 或完成。
# CLI 在一轮里内联串多腿（当场询问用户）；HTTP 把每腿映射为一次请求/一条 SSE 流。
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.types import Command

from .audit import begin_run, end_run, flush_pending
from .schema import ContextSchema

EVENT_TOKEN = "token"
EVENT_TOOL = "tool"
EVENT_NODE = "node"
EVENT_INTERRUPT = "interrupt"
EVENT_FINAL = "final"

# final_answer 可能出现在这两个父图节点的 update 里（与 cli._on_node_update 口径一致）。
_ANSWER_NODES = frozenset({"answer", "sorcery"})
# thread_meta.preview 的截断长度（与 cli.stream_query 的 80 字符口径一致）。
PREVIEW_MAX_CHARS = 80


@dataclass(kw_only=True)
class RunEvent:
    """一条运行事件；``kind`` 取值见模块头部的 EVENT_* 常量。"""

    kind: str
    node: str = ""
    text: str = ""
    data: dict | None = None
    elapsed: float | None = None


def runtime_context_from_env() -> ContextSchema:
    """从环境变量构造 ContextSchema —— 进程边界统一在此注入（CLI/HTTP 共用口径）。"""
    return ContextSchema(
        db_dsn=os.getenv("WLANGGRAPH_POSTGRES_DSN") or None,
        db_dialect=os.getenv("ASKANSWER_DB_DIALECT") or None,
        tenant_id=os.getenv("ASKANSWER_TENANT_ID") or None,
    )


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def query_input(query: str) -> dict:
    """新问题的图输入（与 CLI 完全一致：单条 HumanMessage）。"""
    return {"messages": [HumanMessage(content=query)]}


def resume_input(decision: object) -> Command:
    """interrupt 之后续跑的图输入；decision 形态由各 interrupt 载荷的 type 约定。"""
    return Command(resume=decision)


def preview_of(query: str) -> str | None:
    """thread_meta 的 preview 摘要；空白折叠 + 截断，空串归一为 None。"""
    text = " ".join(str(query or "").split())
    return text[:PREVIEW_MAX_CHARS] or None


class _LegTracker:
    """跨 chunk 维护的一腿小状态：工具规划标记 / 最终答案 / interrupt 载荷 / 节点计时。"""

    def __init__(self) -> None:
        self.in_tool = False
        self.final_answer = ""
        self.interrupt_value: object | None = None
        self._last_finish = time.monotonic()

    def elapsed(self) -> float:
        now = time.monotonic()
        gap = now - self._last_finish
        self._last_finish = now
        return gap


def stream_leg(app, graph_input, *, config, context) -> Iterator[RunEvent]:
    """跑一腿：产出 token/tool/node 事件，最后以 interrupt 或 final 事件收尾。

    不做任何记账（审计/telemetry/元数据）——那是 :func:`run_leg` 的职责。
    """
    tracker = _LegTracker()
    stream = app.stream(
        graph_input, config=config, context=context,
        stream_mode=["updates", "messages"],
    )
    for chunk_mode, payload in stream:
        if chunk_mode == "messages":
            yield from _message_events(payload, tracker)
        else:
            yield from _update_events(payload, tracker)
    if tracker.interrupt_value is None:
        # 兜底：部分 langgraph 版本流结束后不发 __interrupt__，从 state.tasks 反查。
        tracker.interrupt_value = pending_interrupt(app, config)
    if tracker.interrupt_value is not None:
        yield RunEvent(kind=EVENT_INTERRUPT, data=_as_dict(tracker.interrupt_value))
        return
    answer = tracker.final_answer or final_answer_from_state(app, config)
    yield RunEvent(kind=EVENT_FINAL, text=answer)


def run_leg(app, graph_input, *, thread_id, context, preview=None) -> Iterator[RunEvent]:
    """带记账的一腿：审计 begin/flush/end + telemetry 根 span + thread_meta 落库。

    记账口径与 cli.stream_query 的 finally 块一致；front-end 只消费事件。
    HTTP 场景一轮可能拆成多腿（每次 resume 一腿）：审计按腿 flush，同 thread_id 归并；
    resume 腿传 preview=None，upsert_meta 的 COALESCE 语义会保留旧 preview。
    生成器被提前 close（客户端断连）时 finally 仍会执行，记账不丢。
    """
    tokens = begin_run(thread_id, tenant_id=context.tenant_id)
    span = _open_span(thread_id, context.tenant_id)
    config = thread_config(thread_id)
    try:
        yield from stream_leg(app, graph_input, config=config, context=context)
    finally:
        values = _state_values(app, config)
        with suppress(Exception):
            flush_pending(thread_id=thread_id, intent=values.get("intent"))
        _close_span(span)
        with suppress(Exception):
            end_run(tokens)
        _upsert_thread_meta(thread_id, values, context=context, preview=preview)


def _message_events(payload, tracker: _LegTracker) -> Iterator[RunEvent]:
    """messages 通道 → token / tool 事件（口径同 cli._handle_message_chunk）。"""
    if not isinstance(payload, tuple) or len(payload) != 2:
        return
    msg = payload[0]
    if not isinstance(msg, AIMessageChunk):
        return
    if getattr(msg, "tool_call_chunks", None):
        if not tracker.in_tool:  # 每个规划阶段只发一次 tool 事件
            tracker.in_tool = True
            yield RunEvent(kind=EVENT_TOOL, data=_tool_names(msg))
        return
    content = msg.content if isinstance(msg.content, str) else ""
    if not content:
        return
    tracker.in_tool = False
    yield RunEvent(kind=EVENT_TOKEN, text=content)


def _update_events(payload, tracker: _LegTracker) -> Iterator[RunEvent]:
    """updates 通道 → node 事件；顺手捕获 interrupt 载荷与 final_answer。"""
    for node, update in (payload or {}).items():
        if node == "__interrupt__":
            tracker.interrupt_value = extract_interrupt_value(update)
            continue
        if not isinstance(update, dict):
            continue
        if node in _ANSWER_NODES and update.get("final_answer"):
            tracker.final_answer = update["final_answer"]
        yield RunEvent(
            kind=EVENT_NODE, node=node, data=update, elapsed=tracker.elapsed(),
        )


def _tool_names(msg) -> dict:
    """尽力从首个 tool_call_chunk 提取工具名（流式下可能为空，仅供进度展示）。"""
    names = []
    for chunk in getattr(msg, "tool_call_chunks", None) or []:
        name = str(chunk.get("name") or "").strip() if isinstance(chunk, dict) else ""
        if name:
            names.append(name)
    return {"names": names}


def extract_interrupt_value(update):
    """LangGraph 不同版本里 __interrupt__ 的载荷形态不同，做一层兼容。"""
    if isinstance(update, (list, tuple)) and update:
        update = update[0]
    return getattr(update, "value", update)


def pending_interrupt(app, config):
    """从 state.tasks 反查挂起的 interrupt 载荷；无挂起或读取失败返回 None。"""
    try:
        snapshot = app.get_state(config)
    except Exception:
        return None
    for task in getattr(snapshot, "tasks", None) or ():
        interrupts = getattr(task, "interrupts", None) or ()
        if interrupts:
            return getattr(interrupts[0], "value", interrupts[0])
    return None


def final_answer_from_state(app, config) -> str:
    """节点流没给出 final_answer 时的兜底：state.final_answer → 最后一条消息内容。"""
    values = _state_values(app, config)
    answer = values.get("final_answer") or ""
    if answer:
        return answer
    messages = values.get("messages") or []
    if messages:
        content = getattr(messages[-1], "content", "")
        if isinstance(content, str):
            return content
    return ""


def _state_values(app, config) -> dict:
    try:
        state = app.get_state(config)
        return getattr(state, "values", {}) or {}
    except Exception:
        return {}


def _as_dict(value) -> dict:
    """interrupt 载荷统一成 dict，非 dict 载荷包一层（传输层要求可 JSON 化的顶层对象）。"""
    return value if isinstance(value, dict) else {"value": value}


def _open_span(thread_id, tenant_id):
    """开一轮请求的 telemetry 根 span；未启用/失败返回 None（口径同 cli._open_root_span）。"""
    try:
        from . import telemetry

        return telemetry.open_span(
            "askanswer.query", thread_id=thread_id, tenant_id=tenant_id or ""
        )
    except Exception:
        return None


def _close_span(handle) -> None:
    if handle is None:
        return
    try:
        from . import telemetry

        telemetry.close_span(handle)
    except Exception:
        pass


def _upsert_thread_meta(thread_id, values, *, context, preview) -> None:
    """问答后落一行 thread_meta；失败静默（拿到回答比记账更重要，口径同 CLI）。"""
    try:
        from .load import current_model_label
        from .persistence import get_persistence

        messages = values.get("messages") or []
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        get_persistence().upsert_meta(
            thread_id,
            intent=values.get("intent"),
            model_label=current_model_label(),
            preview=preview,
            message_count=human_count,
            tenant_id=context.tenant_id,
        )
    except Exception:
        return
