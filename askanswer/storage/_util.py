"""Small helpers shared by the storage modules (not public API)."""

from __future__ import annotations

import os


def default_tenant_id() -> str | None:
    """写入路径未显式给 tenant 时的兜底：读环境变量，缺省为 None（不分租户）。"""
    return os.environ.get("ASKANSWER_TENANT_ID") or None


def clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def int_or_none(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
