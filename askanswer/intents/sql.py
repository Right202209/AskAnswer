"""SQL intent handler."""

from __future__ import annotations

import re

from ..schema import ContextSchema
from ..state import SearchState
from .base import (
    ClarificationChoice,
    ClarificationRequest,
    EvaluationResult,
    IntentClassification,
    latest_tool_message,
    pass_result,
)

SQL_KEYWORDS = (
    "sql",
    "数据库",
    "数据表",
    "表结构",
    "查询表",
    "查表",
    "建表",
    "postgres",
    "mysql",
    "sqlite",
    "建库",
)
SQL_RE = re.compile(r"(?is)\b(select|insert|update|delete)\b.+\b(from|into|set|where)\b")


class SqlHandler:
    name = "sql"
    priority = 20
    bundle_tags = frozenset({"sql"})
    max_retries = 1

    def local_classify(self, text: str) -> IntentClassification | None:
        clean = str(text or "").strip()
        lowered = clean.lower()
        if any(keyword in lowered for keyword in SQL_KEYWORDS) or SQL_RE.search(clean):
            return IntentClassification(intent=self.name, understanding=clean)
        return None

    def prompt_hint(self, state: SearchState) -> str:
        return "（这是数据库/SQL 类问题；如需查询数据，调用 sql_query 工具。）"

    def clarify(
        self, state: SearchState, context: ContextSchema
    ) -> ClarificationRequest | None:
        """识别为 SQL 但运行时没有数据库连接（DSN）时，让用户决定如何继续。"""
        if context.db_dsn:
            return None  # 已配置连接，直接查库
        # 默认项「仍按数据库处理」= 保持现状（sql_query 会因无连接自行报错并回填），
        # 非 TTY 不回归；TTY 下用户可改选「通用知识作答」把 intent 切成 chat。
        return ClarificationRequest(
            prompt="没检测到数据库连接（未设置 WLANGGRAPH_POSTGRES_DSN）。怎么继续？",
            choices=(
                ClarificationChoice(label="仍按数据库问题处理（无连接可能失败）"),
                ClarificationChoice(
                    label="改用通用知识作答（不连数据库）",
                    updates={"intent": "chat"},
                ),
            ),
            default_index=0,
        )

    def evaluate(self, state: SearchState) -> EvaluationResult:
        tool_message = latest_tool_message(state, "sql_query")
        if tool_message is None:
            return pass_result("no sql_query result to evaluate")
        content = str(getattr(tool_message, "content", "") or "")
        empty_markers = ("空结果", "未返回结果", "no rows", "0 rows", "empty result")
        if any(marker in content.lower() for marker in empty_markers):
            return EvaluationResult(
                decision="retry",
                retry_directive={
                    "instruction": "sql_query 返回空结果。请检查表名、过滤条件或先查询表结构后再重试。",
                },
                reason="sql result appears empty",
            )
        return pass_result("sql result looks usable")

    def cli_label(self, update: dict) -> str:
        return "sql"
