"""LangSmith exporter：把 telemetry span/事件映射成 LangSmith run。

仅在 ``langsmith`` SDK 可导入且 ``LANGSMITH_API_KEY`` / ``LANGCHAIN_API_KEY`` 存在时
构建；任何一步失败都返回 None，让上层安静退回“未启用”。写入全部 best-effort、
包在 try/except 里 —— 可观测性绝不能拖垮主流程，也要容忍 SDK 版本间的签名差异。
"""

from __future__ import annotations

import os
import uuid
from typing import Any


def build_langsmith_exporter():
    """构建 LangSmith exporter；SDK 缺失 / client 初始化失败时返回 None。"""
    try:
        from langsmith import Client
    except ImportError:
        return None
    try:
        client = Client()
    except Exception:
        # 缺 endpoint / key 校验失败等：直接放弃该 exporter。
        return None
    project = os.environ.get("LANGSMITH_PROJECT") or os.environ.get(
        "LANGCHAIN_PROJECT"
    ) or "askanswer"
    return _LangSmithExporter(client, project)


class _LangSmithExporter:
    """span_id → LangSmith run 的薄适配层。"""

    def __init__(self, client, project: str):
        self._client = client
        self._project = project
        # span_id → run uuid，用来建立 parent/child run 关系。
        self._runs: dict[str, uuid.UUID] = {}

    def start_span(self, span_id: str, name: str, parent_id: str | None, attrs: dict) -> None:
        run_id = uuid.uuid4()
        self._runs[span_id] = run_id
        parent_run_id = self._runs.get(parent_id) if parent_id else None
        try:
            self._client.create_run(
                name=name,
                run_type="chain",
                inputs=dict(attrs or {}),
                id=run_id,
                parent_run_id=parent_run_id,
                project_name=self._project,
            )
        except Exception:
            # 版本差异 / 网络问题：丢弃这次 run，不影响主流程。
            self._runs.pop(span_id, None)

    def end_span(self, span_id: str, duration_ms: int, error: str | None) -> None:
        run_id = self._runs.pop(span_id, None)
        if run_id is None:
            return
        try:
            self._client.update_run(
                run_id,
                error=error,
                outputs={"duration_ms": duration_ms},
            )
        except Exception:
            pass

    def emit(self, event: dict[str, Any]) -> None:
        # 离散事件（llm_call / tool_call）折叠进当前 span 的一次短命 child run，
        # 让 token / tool 名在 LangSmith 上也能看到；无 parent 时跳过。
        parent_span = event.get("span_id")
        parent_run_id = self._runs.get(parent_span) if parent_span else None
        if parent_run_id is None:
            return
        run_id = uuid.uuid4()
        try:
            self._client.create_run(
                name=event.get("kind") or "event",
                run_type="llm" if event.get("kind") == "llm_call" else "tool",
                inputs={"tool_name": event.get("tool_name")},
                outputs={
                    "input_tokens": event.get("input_tokens"),
                    "output_tokens": event.get("output_tokens"),
                    "duration_ms": event.get("duration_ms"),
                },
                error=event.get("error"),
                id=run_id,
                parent_run_id=parent_run_id,
                project_name=self._project,
            )
        except Exception:
            pass
