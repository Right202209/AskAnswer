"""Intent handler protocol and shared classification helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field, model_validator

from ..state import SearchState

Decision = Literal["pass", "retry"]


@dataclass(frozen=True)
class EvaluationResult:
    decision: Decision
    retry_directive: dict | None = None
    reason: str = ""


@dataclass(frozen=True)
class ClarificationChoice:
    """一个澄清选项：``label`` 展示在菜单里，被选中后 ``updates`` 并入 SearchState。"""
    label: str
    updates: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClarificationRequest:
    """intent handler 在 answer 之前请求用户澄清的结构化载荷。

    约定（保证非回归）：``choices[default_index]`` 必须等价于「保持现状」——
    非 TTY 环境直接取默认项，行为与未接入澄清时完全一致；澄清只在交互式下增益。
    ``free_text_field`` 非空时，菜单追加一个「手动输入」项，命中后把用户键入的
    文本写到该 SearchState 字段（留空则视作放弃、保持现状）。
    """
    prompt: str
    choices: tuple[ClarificationChoice, ...]
    default_index: int = 0
    free_text_field: str = ""
    free_text_label: str = "其他（手动输入）"
    free_text_prompt: str = "请输入："


class IntentClassification(BaseModel):
    intent: str = Field(default="search")
    file_path: str = ""
    search_query: str = ""
    understanding: str = ""

    @model_validator(mode="after")
    def _normalize_strings(self) -> "IntentClassification":
        self.intent = str(self.intent or "").lower().strip()
        self.file_path = str(self.file_path or "").strip()
        self.search_query = str(self.search_query or "").strip()
        self.understanding = str(self.understanding or "").strip()
        return self


class IntentHandler(Protocol):
    name: str
    priority: int
    bundle_tags: frozenset[str]
    max_retries: int

    def local_classify(self, text: str) -> IntentClassification | None: ...

    def prompt_hint(self, state: SearchState) -> str: ...

    def evaluate(self, state: SearchState) -> EvaluationResult: ...

    def cli_label(self, update: dict) -> str: ...

    # 可选能力：handler 可另外实现
    #   ``clarify(self, state, context) -> ClarificationRequest | None``
    # 在 answer 之前请求用户澄清缺失/含糊的信息（缺路径、缺 DSN、范围不清等）。
    # 不在协议里强制，未实现即视为「无需澄清」——见 ``get_clarification``。


FILE_EXTENSIONS = (
    "txt", "md", "markdown", "rst",
    "json", "jsonl", "yaml", "yml", "toml", "ini", "cfg", "conf", "env",
    "csv", "tsv", "xlsx", "xls",
    "pdf", "docx", "pptx",
    "html", "htm", "xml", "svg",
    "py", "pyi", "ipynb",
    "js", "jsx", "ts", "tsx",
    "go", "rs", "java", "kt", "swift", "rb", "php",
    "c", "h", "cpp", "hpp", "cc", "cs",
    "sh", "bash", "zsh", "ps1",
    "sql", "log",
)

FILE_PATH_RE = re.compile(
    r"""(?ix)
    (?:
        ["'`“”‘’]?
        (
            (?:[a-z]:[\\/]|\.{1,2}[\\/]|[\\/]|~/)?
            [^\s"'`“”‘’<>|]+
            \.(?:%s)
        )
        ["'`“”‘’]?
    )
    """ % "|".join(FILE_EXTENSIONS)
)


def extract_file_path(text: str) -> str:
    match = FILE_PATH_RE.search(str(text or ""))
    if not match:
        return ""
    return match.group(1).strip().strip("\"'`“”‘’")


def latest_tool_message(state: SearchState, tool_name: str | None = None) -> ToolMessage | None:
    for message in reversed(state.get("messages") or []):
        if not isinstance(message, ToolMessage):
            continue
        if tool_name is None or getattr(message, "name", None) == tool_name:
            return message
    return None


def pass_result(reason: str = "") -> EvaluationResult:
    return EvaluationResult(decision="pass", reason=reason)


def get_clarification(
    handler: IntentHandler,
    state: SearchState,
    context: Any,
) -> ClarificationRequest | None:
    """取当前 handler 的澄清请求；未实现 ``clarify`` 或其抛异常都安全降级为 None。

    澄清是「锦上添花」，任何失败都不该阻断正常回答，因此这里吞掉异常返回 None。
    """
    clarify = getattr(handler, "clarify", None)
    if clarify is None:
        return None
    try:
        return clarify(state, context)
    except Exception:
        return None
