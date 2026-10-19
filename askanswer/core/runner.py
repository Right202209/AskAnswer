"""UI-agnostic graph runner: one "leg" as a stream of :class:`RunEvent`.

A *leg* runs from one graph input (a new question or a resume decision) until the
graph either interrupts for human input or completes.  Event contract
(事件契约，宿主可按需渲染或编码):

- ``token``     – incremental user-facing LLM text (面向用户的增量文本);
- ``tool``      – the LLM started planning tool calls; consumers should clear their
                  token buffer, planning text is not part of the answer;
- ``node``      – a graph node finished (``data`` is the raw update dict and may
                  contain message objects; hosts serialise as needed);
- ``interrupt`` – the graph is suspended waiting for human input (``data`` is the
                  interrupt payload; continue with :func:`resume_input`);
- ``final``     – the leg completed normally (``text`` is the final answer).

:func:`stream_leg` only yields events.  :func:`run_leg` adds bookkeeping: the audit
buffer, a telemetry root span and the ``thread_meta`` upsert, written to the
persistence backend resolved from the app (``NullPersistence`` for in-memory
checkpointers).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.types import Command

from ..config.schema import ContextSchema
from ..config.settings import ensure_bootstrapped, resolve_db_dsn
from ..observability.audit import begin_run, end_run, flush_pending

EVENT_TOKEN = "token"
EVENT_TOOL = "tool"
EVENT_NODE = "node"
EVENT_INTERRUPT = "interrupt"
EVENT_FINAL = "final"

# final_answer 可能出现在这两个父图节点的 update 里。
_ANSWER_NODES = frozenset({"answer", "sorcery"})
# thread_meta.preview 的截断长度。
PREVIEW_MAX_CHARS = 80


@dataclass(kw_only=True)
class RunEvent:
    """One run event; ``kind`` is one of the ``EVENT_*`` constants (见模块头部)."""

    kind: str
    node: str = ""
    text: str = ""
    data: dict | None = None
    elapsed: float | None = None


def runtime_context_from_env() -> ContextSchema:
    """Build a :class:`ContextSchema` from the environment (bootstraps config lazily).

    Reads ``ASKANSWER_DB_DSN`` (``WLANGGRAPH_POSTGRES_DSN`` as a deprecated fallback),
    ``ASKANSWER_DB_DIALECT`` and ``ASKANSWER_TENANT_ID``.
    """
    ensure_bootstrapped()
    return ContextSchema(
        db_dsn=resolve_db_dsn(),
        db_dialect=os.getenv("ASKANSWER_DB_DIALECT") or None,
        tenant_id=os.getenv("ASKANSWER_TENANT_ID") or None,
    )


def thread_config(thread_id: str) -> dict:
    """LangGraph config selecting the checkpoint thread ``thread_id``."""
    return {"configurable": {"thread_id": thread_id}}


def query_input(query: str) -> dict:
    """Graph input for a new question: a single ``HumanMessage`` (新问题的图输入)."""
    return {"messages": [HumanMessage(content=query)]}


def resume_input(decision: object) -> Command:
    """Graph input that resumes after an interrupt.

    The ``decision`` shape depends on the interrupt payload's ``type``:
    ``{"approve": bool}`` for ``confirm_*`` (shell may also pass ``"command"``),
    ``{"index": int, "text": str}`` for ``clarify``.
    """
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
    """Run one leg and yield ``token`` / ``tool`` / ``node`` events, ending with
    exactly one ``interrupt`` or ``final`` event.

    No bookkeeping (audit / telemetry / thread metadata) — that is :func:`run_leg`'s job.
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


def run_leg(
    app,
    graph_input,
    *,
    thread_id,
    context,
    preview=None,
    persistence=None,
) -> Iterator[RunEvent]:
    """Run one leg with bookkeeping: audit buffer, telemetry root span, thread_meta.

    (带记账的一腿：审计 begin/flush/end + telemetry 根 span + thread_meta 落库。)

    宿主只需消费事件；一轮可能因确认/澄清拆成多段（每次 resume 一段），审计按段
    flush，并按同一 thread_id 归并；
    resume 腿传 preview=None，upsert_meta 的 COALESCE 语义会保留旧 preview。
    生成器被提前 close（客户端断连）时 finally 仍会执行，记账不丢。

    ``persistence``：审计与 thread_meta 的写入后端。缺省时按 ``app.checkpointer``
    解析——它是某个 ``PersistenceManager`` 的 saver 时写入该 manager，否则（如
    ``InMemorySaver``）使用 ``NullPersistence``，纯内存运行不会创建任何 SQLite 文件。
    """
    backend = resolve_persistence(app, persistence)
    tokens = begin_run(thread_id, tenant_id=context.tenant_id, persistence=backend)
    span = _open_span(thread_id, context.tenant_id)
    config = thread_config(thread_id)
    try:
        yield from stream_leg(app, graph_input, config=config, context=context)
    finally:
        values = _state_values(app, config)
        with suppress(Exception):
            flush_pending(
                thread_id=thread_id, intent=values.get("intent"), persistence=backend
            )
        _close_span(span)
        with suppress(Exception):
            end_run(tokens)
        _upsert_thread_meta(
            backend, thread_id, values, context=context, preview=preview
        )


def resolve_persistence(app, explicit=None):
    """Pick the persistence backend ``run_leg`` writes audit / thread_meta to.

    Order: ``explicit`` → the live ``PersistenceManager`` whose saver is
    ``app.checkpointer`` → ``NullPersistence``.  Never opens a database itself.
    """
    if explicit is not None:
        return explicit
    from ..storage import NULL_PERSISTENCE, persistence_for_checkpointer

    manager = persistence_for_checkpointer(getattr(app, "checkpointer", None))
    return manager if manager is not None else NULL_PERSISTENCE


def _message_events(payload, tracker: _LegTracker) -> Iterator[RunEvent]:
    """messages 通道 → token / tool 事件。"""
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
    """开一轮请求的 telemetry 根 span；未启用或失败时返回 None。"""
    try:
        from ..observability import telemetry

        return telemetry.open_span(
            "askanswer.query", thread_id=thread_id, tenant_id=tenant_id or ""
        )
    except Exception:
        return None


def _close_span(handle) -> None:
    if handle is None:
        return
    try:
        from ..observability import telemetry

        telemetry.close_span(handle)
    except Exception:
        pass


def _upsert_thread_meta(backend, thread_id, values, *, context, preview) -> None:
    """问答后落一行 thread_meta；失败静默，避免记账影响回答。"""
    try:
        from ..llm.proxy import current_model_label

        messages = values.get("messages") or []
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        backend.upsert_meta(
            thread_id,
            intent=values.get("intent"),
            model_label=current_model_label(),
            preview=preview,
            message_count=human_count,
            tenant_id=context.tenant_id,
        )
    except Exception:
        return
