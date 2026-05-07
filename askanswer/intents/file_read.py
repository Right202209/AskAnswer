"""File-read intent handler."""

from __future__ import annotations

from ..state import SearchState
from .base import (
    EvaluationResult,
    IntentClassification,
    extract_file_path,
    latest_tool_message,
    pass_result,
)


FILE_READ_KEYWORDS = (
    "读",
    "读取",
    "分析",
    "打开",
    "查看",
    "文件",
    "read",
    "analyze",
    "open",
    "view",
    "file",
)


class FileReadHandler:
    name = "file_read"
    priority = 10
    bundle_tags = frozenset({"file_read"})
    max_retries = 1

    def local_classify(self, text: str) -> IntentClassification | None:
        clean = str(text or "").strip()
        lowered = clean.lower()
        file_path = extract_file_path(clean)
        if file_path and any(word in lowered for word in FILE_READ_KEYWORDS):
            return IntentClassification(
                intent=self.name,
                file_path=file_path,
                understanding=f"读取或分析本地文件：{file_path}",
            )
        return None

    def prompt_hint(self, state: SearchState) -> str:
        file_path = state.get("file_path") or ""
        if file_path:
            return f"（这是读文件请求，请调用 read_file 工具读取 `{file_path}` 后再作答。）"
        return "（这是读文件请求，请调用 read_file 工具读取目标文件后再作答。）"

    def evaluate(self, state: SearchState) -> EvaluationResult:
        tool_message = latest_tool_message(state, "read_file")
        if tool_message is None:
            return pass_result("no read_file result to evaluate")
        content = str(getattr(tool_message, "content", "") or "")
        failure_markers = ("执行失败", "读取失败", "No such file", "not found", "不存在")
        if any(marker in content for marker in failure_markers):
            return EvaluationResult(
                decision="retry",
                retry_directive={
                    "instruction": "read_file 工具读取失败。请根据用户原文重新确认路径，必要时先调用 pwd，再重试 read_file。",
                    "file_path": state.get("file_path", ""),
                },
                reason="read_file failed",
            )
        return pass_result("read_file result looks usable")

    def cli_label(self, update: dict) -> str:
        file_path = update.get("file_path", "")
        return f"file_read: {file_path}" if file_path else "file_read"
