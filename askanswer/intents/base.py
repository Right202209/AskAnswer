"""Intent handler protocol and shared classification helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field, model_validator

from ..state import SearchState


Decision = Literal["pass", "retry"]


@dataclass(frozen=True)
class EvaluationResult:
    decision: Decision
    retry_directive: dict | None = None
    reason: str = ""


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
