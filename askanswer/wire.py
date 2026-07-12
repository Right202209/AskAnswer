# HTTP 传输层小工具：SSE 帧编码、JSON 安全裁剪、请求侧校验。
#
# 职责边界：runner 产出"忠实"事件（node 事件携带原始 update dict，含消息对象等
# 非 JSON 值）；本模块负责传输层裁剪 —— node 只保留标量摘要，interrupt 载荷做
# 递归 JSON 化（interrupt 载荷本就是给用户看的，str() 兜底不构成泄露面扩大）。
# 请求侧校验（路径解析 / Origin 判定 / thread_id 格式）也集中在此，供 server 及
# 后续 C4 的只读 JSON 端点复用。
from __future__ import annotations

import json
import re
from http import HTTPStatus
from urllib.parse import parse_qs, urlparse

from .runner import (
    EVENT_FINAL,
    EVENT_INTERRUPT,
    EVENT_NODE,
    EVENT_TOKEN,
    EVENT_TOOL,
    RunEvent,
)

# 递归 JSON 化的最大深度；超过后降级为 str()，防御自引用/超深结构。
MAX_JSON_DEPTH = 6
# node 摘要里字符串字段的截断长度（完整答案走 final 事件，这里只是进度展示）。
MAX_SUMMARY_CHARS = 200
_ELAPSED_DECIMALS = 3

THREAD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class RequestError(Exception):
    """带 HTTP 状态码的请求级错误；message 必须是可直接回给客户端的安全文案。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


# ---- SSE / JSON 编码 --------------------------------------------------------
def sse_frame(event: str, data) -> bytes:
    """一条 SSE 帧：``event: <name>\\ndata: <json>\\n\\n``。

    json.dumps 不会输出裸换行（换行都转义成 \\n），因此 data 恒为单行、无需拆多行。
    """
    body = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {body}\n\n".encode()


def event_wire(event: RunEvent) -> tuple[str, dict]:
    """RunEvent → (SSE 事件名, 可 JSON 化载荷)。"""
    if event.kind == EVENT_TOKEN:
        return EVENT_TOKEN, {"text": event.text}
    if event.kind == EVENT_TOOL:
        return EVENT_TOOL, {"names": list((event.data or {}).get("names") or [])}
    if event.kind == EVENT_NODE:
        return EVENT_NODE, _node_wire(event)
    if event.kind == EVENT_INTERRUPT:
        return EVENT_INTERRUPT, json_safe(event.data or {})
    if event.kind == EVENT_FINAL:
        return EVENT_FINAL, {"text": event.text}
    return event.kind or "message", json_safe(event.data or {})


def json_safe(value, depth: int = MAX_JSON_DEPTH):
    """递归转成 JSON 可编码值；未知对象与超深层级降级为 str()。"""
    if depth <= 0:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v, depth - 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v, depth - 1) for v in value]
    return str(value)


def _node_wire(event: RunEvent) -> dict:
    data = {"node": event.node, "summary": _scalar_summary(event.data or {})}
    if event.elapsed is not None:
        data["elapsed"] = round(event.elapsed, _ELAPSED_DECIMALS)
    return data


def _scalar_summary(update: dict) -> dict:
    """只保留 update 里的标量字段（messages/pending_* 等复杂对象一律丢弃）。"""
    summary = {}
    for key, value in update.items():
        if isinstance(value, (bool, int, float)):
            summary[key] = value
        elif isinstance(value, str):
            summary[key] = value[:MAX_SUMMARY_CHARS]
    return summary


# ---- 请求侧校验 --------------------------------------------------------------
def split_path(raw: str) -> tuple[str, dict]:
    """请求行 path → (规整路径, 查询参数 dict)；尾部斜杠归一。"""
    parsed = urlparse(raw)
    return (parsed.path.rstrip("/") or "/"), parse_qs(parsed.query)


def is_local_origin(origin: str) -> bool:
    """Origin 头是否指向本机（跨源浏览器请求的 CSRF 闸门）。"""
    host = (urlparse(origin).hostname or "").lower()
    return host in LOCAL_HOSTS


def normalize_thread_id(value) -> str | None:
    """thread_id 清洗：空→None；非法字符/超长→RequestError(400)。"""
    text = str(value or "").strip()
    if not text:
        return None
    if not THREAD_ID_RE.match(text):
        raise RequestError(HTTPStatus.BAD_REQUEST, "invalid thread_id format")
    return text
