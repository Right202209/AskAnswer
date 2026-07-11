"""三路线可观测性分层：本地 audit / LangSmith / OpenTelemetry 各走独立写入路径。

设计不变量（对齐 execution-plan Phase 2.2）：
- **不污染 SearchState**：trace 上下文全部走 contextvars，节点入口 push、出口 pop，
  绝不写进 state（否则会膨胀 checkpoint 并破坏状态语义）。
- **关闭即零开销**：环境变量未开启时不 import 任何 SDK、不建 client；``emit_event`` /
  ``span`` 退化成一次列表真值判断后立即返回。
- **不缓存 ``_ModelProxy``**：LLM 回调只在 ``on_llm_end`` 读取响应，从不持有 model
  引用，避免 ``/model`` 热替换后指向旧 backend。
- **与 audit 共享最小事件 schema（kind / tool_name / duration_ms / tokens），写入路径独立**。

启用开关：
- ``LANGSMITH_API_KEY`` / ``LANGCHAIN_API_KEY`` → LangSmith exporter；
- ``ASKANSWER_OTEL_EXPORTER`` → OpenTelemetry exporter。
SDK 未安装时对应 exporter 静默跳过（build 返回 None）。
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from langchain_core.callbacks import BaseCallbackHandler


# 已激活的 exporter 列表；空列表 = 未启用（emit/span 走零开销分支）。
_exporters: list = []
_initialized = False

# 当前 span 栈（span_id 序列）。给子 span 关联 parent，且跨 interrupt 不持久化。
_SPAN_STACK: ContextVar[tuple] = ContextVar("askanswer_telemetry_span_stack", default=())


def init_telemetry() -> None:
    """按环境变量装配 exporter；重复调用幂等。CLI 启动时调一次即可。"""
    global _initialized, _exporters
    if _initialized:
        return
    _initialized = True
    _exporters = _load_exporters()


def _load_exporters() -> list:
    """按环境变量惰性构建各 exporter；SDK 缺失或初始化失败时跳过。"""
    exporters: list = []
    if os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"):
        from .langsmith import build_langsmith_exporter

        exporter = build_langsmith_exporter()
        if exporter is not None:
            exporters.append(exporter)
    if os.environ.get("ASKANSWER_OTEL_EXPORTER"):
        from .otel import build_otel_exporter

        exporter = build_otel_exporter()
        if exporter is not None:
            exporters.append(exporter)
    return exporters


def is_enabled() -> bool:
    return bool(_exporters)


def emit_event(
    *,
    kind: str,
    tool_name: str | None = None,
    duration_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    model_label: str | None = None,
    error: str | None = None,
) -> None:
    """把一条离散事件（tool_call / llm_call）派发给所有 exporter。未启用时零开销。"""
    if not _exporters:
        return
    stack = _SPAN_STACK.get()
    event = {
        "kind": kind,
        "tool_name": tool_name,
        "duration_ms": duration_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_label": model_label,
        "error": error,
        "span_id": stack[-1] if stack else None,
    }
    for exporter in _exporters:
        try:
            exporter.emit(event)
        except Exception:
            # 可观测性绝不能拖垮主流程。
            pass


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    """一段 trace span（``with`` 形式）：入口 push、出口带 duration 收尾。

    未启用时直接 ``yield`` —— 不生成 id、不取时间、不 push contextvar。
    """
    handle = open_span(name, **attrs)
    if handle is None:
        yield
        return
    error: str | None = None
    try:
        yield
    except Exception as exc:  # 记录后重新抛出，不吞异常
        error = str(exc)
        raise
    finally:
        close_span(handle, error)


def open_span(name: str, **attrs: Any):
    """手动开一个 span（与 ``audit.begin_run`` 同风格），返回 handle 或 None。

    供 ``stream_query`` 这类“手动 begin/finally end”的调用点使用；未启用返回
    None，调用方对 None 做 close 时是 no-op。
    """
    if not _exporters:
        return None
    span_id = uuid.uuid4().hex
    stack = _SPAN_STACK.get()
    parent_id = stack[-1] if stack else None
    token = _SPAN_STACK.set(stack + (span_id,))
    for exporter in _exporters:
        try:
            exporter.start_span(span_id, name, parent_id, attrs)
        except Exception:
            pass
    return (span_id, token, time.monotonic())


def close_span(handle, error: str | None = None) -> None:
    """关闭 ``open_span`` 返回的 handle；handle 为 None 时 no-op。"""
    if handle is None:
        return
    span_id, token, started = handle
    duration_ms = int((time.monotonic() - started) * 1000)
    for exporter in _exporters:
        try:
            exporter.end_span(span_id, duration_ms, error)
        except Exception:
            pass
    _SPAN_STACK.reset(token)


def llm_callback(model_label: str):
    """启用时返回一个 LangChain 回调，把 LLM 调用发到 telemetry；否则返回 None。

    ``load.py`` 在注入 audit 回调的同一处按需追加它，从而所有经 ``model`` 的
    LLM 调用（answer / understand / sorcery / sql / helix）都被自动 trace。
    """
    if not _exporters:
        return None
    return _TelemetryLLMCallback(model_label=model_label)


class _TelemetryLLMCallback(BaseCallbackHandler):
    """把单次 LLM 调用的时延/token 作为 telemetry 事件发出（独立于 audit 路径）。"""

    def __init__(self, *, model_label: str):
        self.model_label = model_label
        self.started_at = time.monotonic()

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.started_at = time.monotonic()

    def on_llm_start(self, *args, **kwargs) -> None:
        self.started_at = time.monotonic()

    def on_llm_end(self, response, **kwargs) -> None:
        input_tokens, output_tokens = _extract_tokens(response)
        emit_event(
            kind="llm_call",
            model_label=self.model_label,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=int((time.monotonic() - self.started_at) * 1000),
        )

    def on_llm_error(self, error, **kwargs) -> None:
        emit_event(
            kind="llm_call",
            model_label=self.model_label,
            duration_ms=int((time.monotonic() - self.started_at) * 1000),
            error=str(error),
        )


def _extract_tokens(response) -> tuple[int | None, int | None]:
    """从 LLM 响应里尽力取 (input_tokens, output_tokens)；取不到返回 (None, None)。

    只认现代 LangChain 的 ``usage_metadata`` 标准字段，保持与 audit 的 token
    抽取相互独立（写入路径解耦）。
    """
    for group in getattr(response, "generations", None) or ():
        for generation in group or ():
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) if message else None
            if isinstance(usage, dict):
                return _int_or_none(usage.get("input_tokens")), _int_or_none(
                    usage.get("output_tokens")
                )
    return None, None


def _int_or_none(value) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
