# 斜杠命令的共享底座：租户口径、``/threads`` 列表缓存、thread 解析、参数解析。
#
# 放在叶子模块里（不 import 任何 commands 子模块），供所有命令模块复用而不形成环。
from __future__ import annotations

import os
import shlex
import time

from ...persistence import ThreadMeta, get_persistence
from ..render import render_error

# 缓存最近一次 ``/threads`` 的结果，``/resume <序号>`` / ``/delete <序号>`` 据此寻址。
# REPL 是单线程的，不需要锁。
_LAST_LIST: list[ThreadMeta] = []


def remember_threads(threads: list[ThreadMeta]) -> None:
    """记住最近一次 ``/threads`` 的输出，供按序号寻址。"""
    global _LAST_LIST
    _LAST_LIST = threads


def forget_thread(thread_id: str) -> None:
    """从 ``/threads`` 缓存里剔除一条（删除后避免 /resume 误中）。"""
    global _LAST_LIST
    _LAST_LIST = [m for m in _LAST_LIST if m.thread_id != thread_id]


def _current_tenant() -> str | None:
    """当前 CLI 会话归属的租户；未设 ``ASKANSWER_TENANT_ID`` 时为 None（不分租户）。

    所有按租户过滤的命令（/threads、/audit、/usage、/resume、/delete …）都以它作为
    persistence 调用的 ``tenant_id``，让两个 tenant 在同一 SQLite 文件下互不可见。
    """
    return os.getenv("ASKANSWER_TENANT_ID") or None


def _resolve_thread(arg: str) -> ThreadMeta | None:
    """把 ``/resume 1`` 或 ``/resume <id 前缀>`` 解析为 ``ThreadMeta``。

    解析顺序：
    1. 纯数字 → 视作针对最近一次 ``/threads`` 列表的序号；
    2. 完整 ID 精确匹配；
    3. 4 字符及以上前缀匹配（多于一条则返回 None 让上层提示歧义）。
    """
    arg = (arg or "").strip()
    if not arg:
        return None
    pm = get_persistence()
    tenant = _current_tenant()

    # 1) 序号：相对最近一次 /threads 的输出
    if arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(_LAST_LIST):
            return _LAST_LIST[idx]
        return None

    # 2) 完整匹配（UUID 是 36 字符，但允许任意完整 ID）；越 tenant 访问会被 get_meta 拦掉
    meta = pm.get_meta(arg, tenant_id=tenant)
    if meta is not None:
        return meta

    # 3) 前缀匹配：≥4 字符才生效，避免 "a" 这种过宽匹配
    matches = pm.find_by_prefix(arg, limit=2, tenant_id=tenant)
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_thread_or_current(arg: str | None, current: str) -> ThreadMeta | None:
    token = (arg or "current").strip()
    if token.lower() in {"", "current", "this", "."}:
        # 当前会话不做 tenant 校验：它就是本进程正在跑的 thread，天然属于本 tenant。
        meta = get_persistence().get_meta(current)
        if meta is not None:
            return meta
        return ThreadMeta(
            thread_id=current,
            title=None,
            created_at=int(time.time()),
            updated_at=int(time.time()),
        )
    return _resolve_thread(token)


def _has_pending_interrupt(app, thread_id: str) -> bool:
    """检测某 thread 是否还有挂起的 ``interrupt()`` 任务（HITL 确认没收尾）。"""
    if app is None:
        return False
    try:
        snap = app.get_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        return False
    for task in (getattr(snap, "tasks", None) or ()):
        if getattr(task, "interrupts", None):
            return True
    return False


def _split_args(args: str) -> list[str] | None:
    try:
        return shlex.split(args)
    except ValueError as exc:
        render_error(f"参数解析失败: {exc}")
        return None


def _parse_nonnegative_int(raw: str) -> int | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None
