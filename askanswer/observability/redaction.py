"""Audit-argument redaction.

Values under sensitive keys (``api_key``, ``token``, ``password`` …) are masked as a
whole and e-mail addresses inside string values are masked, before tool args are
written to ``audit_event``.  The confirmation UI still shows the raw values: the
user must approve what will actually run.
"""

from __future__ import annotations

import re
from typing import Any

# 审计脱敏：命中这些 key 的参数值不落库（UI 确认框仍显示原值，用户需据实批准）。
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|key|token|secret|password|passwd|authorization|auth|credential)s?$",
    re.IGNORECASE,
)
# 参数值里出现的 email 一律打码（execution-plan 3.5：审计去 token/email 等）。
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_REDACTED = "***"
_MAX_REDACT_DEPTH = 4


def redact_audit_args(args: Any, _depth: int = 0) -> Any:
    """递归脱敏审计参数：敏感 key 的值整体打码，字符串值内的 email 打码。"""
    if _depth >= _MAX_REDACT_DEPTH:
        return _REDACTED
    if isinstance(args, dict):
        return {
            key: _REDACTED
            if _SENSITIVE_KEY.search(str(key))
            else redact_audit_args(value, _depth + 1)
            for key, value in args.items()
        }
    if isinstance(args, (list, tuple)):
        return [redact_audit_args(item, _depth + 1) for item in args]
    if isinstance(args, str):
        return _EMAIL.sub(_REDACTED, args)
    return args
