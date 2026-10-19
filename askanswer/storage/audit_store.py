"""Audit-event storage (``audit_event``): batch inserts, listing and usage summaries.

Mixin over the shared connection owned by
:class:`askanswer.storage.manager.PersistenceManager`.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from ._util import clip as _clip
from ._util import default_tenant_id as _default_tenant_id
from ._util import int_or_none as _int_or_none


@dataclass
class AuditEvent:
    """``audit_event`` 表的一行。"""

    id: int
    thread_id: str
    ts: int
    kind: str
    tool_name: str | None = None
    args_summary: str | None = None
    result_size: int | None = None
    model_label: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int | None = None
    intent: str | None = None
    error: str | None = None
    tenant_id: str | None = None
    cached_tokens: int | None = None


class AuditStoreMixin:
    """``audit_event`` 写入与查询。"""

    _conn: sqlite3.Connection

    def log_audit_event(
        self,
        thread_id: str,
        *,
        kind: str,
        ts: int | None = None,
        tool_name: str | None = None,
        args_summary: str | None = None,
        result_size: int | None = None,
        model_label: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        duration_ms: int | None = None,
        intent: str | None = None,
        error: str | None = None,
        tenant_id: str | None = None,
        cached_tokens: int | None = None,
    ) -> None:
        """追加一条审计事件。``tenant_id`` 缺省时回退到环境变量。"""
        if not thread_id or not kind:
            return
        tenant = tenant_id if tenant_id is not None else _default_tenant_id()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO audit_event(
                    thread_id, ts, kind, tool_name, args_summary, result_size,
                    model_label, input_tokens, output_tokens, duration_ms,
                    intent, error, tenant_id, cached_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    int(ts or time.time()),
                    kind,
                    tool_name,
                    _clip(args_summary, 200) if args_summary is not None else None,
                    result_size,
                    model_label,
                    input_tokens,
                    output_tokens,
                    duration_ms,
                    intent,
                    _clip(error, 200) if error is not None else None,
                    tenant,
                    _int_or_none(cached_tokens) or 0,
                ),
            )

    def log_audit_events(self, events: list[dict]) -> int:
        """批量写入审计事件。返回成功写入条数。

        相比 N 次 ``log_audit_event``，这里只取一次锁、跑一个事务，
        热路径（每个 LLM/tool 调用一条）总开销显著降低。
        每条事件的 ``tenant_id`` 缺省时回退到环境变量。
        """
        rows = []
        default_tenant = _default_tenant_id()
        for event in events:
            thread_id = event.get("thread_id")
            kind = event.get("kind")
            if not thread_id or not kind:
                continue
            args_summary = event.get("args_summary")
            error = event.get("error")
            tenant = event.get("tenant_id")
            if tenant is None:
                tenant = default_tenant
            rows.append((
                thread_id,
                int(event.get("ts") or time.time()),
                kind,
                event.get("tool_name"),
                _clip(args_summary, 200) if args_summary is not None else None,
                event.get("result_size"),
                event.get("model_label"),
                event.get("input_tokens"),
                event.get("output_tokens"),
                event.get("duration_ms"),
                event.get("intent"),
                _clip(error, 200) if error is not None else None,
                tenant,
                _int_or_none(event.get("cached_tokens")) or 0,
            ))
        if not rows:
            return 0
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO audit_event(
                    thread_id, ts, kind, tool_name, args_summary, result_size,
                    model_label, input_tokens, output_tokens, duration_ms,
                    intent, error, tenant_id, cached_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def list_audit_events(
        self,
        *,
        thread_id: str | None = None,
        limit: int = 50,
        kind: str | None = None,
        days: int | None = None,
        tenant_id: str | None = None,
    ) -> list[AuditEvent]:
        """按时间倒序列出审计事件。``tenant_id`` 非 None 时按租户过滤。"""
        sql = (
            "SELECT id, thread_id, ts, kind, tool_name, args_summary, result_size, "
            "model_label, input_tokens, output_tokens, duration_ms, intent, error, "
            "tenant_id, cached_tokens "
            "FROM audit_event "
        )
        where: list[str] = []
        params: list = []
        if thread_id:
            where.append("thread_id = ?")
            params.append(thread_id)
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if days and days > 0:
            where.append("ts >= ?")
            params.append(int(time.time()) - int(days) * 86400)
        if tenant_id is not None:
            where.append("tenant_id = ?")
            params.append(tenant_id)
        if where:
            sql += "WHERE " + " AND ".join(where) + " "
        sql += "ORDER BY ts DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_audit(r) for r in rows]

    def usage_summary(
        self,
        *,
        thread_id: str | None = None,
        days: int | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, list[dict]]:
        """返回模型与工具两个维度的轻量聚合。``tenant_id`` 非 None 时按租户过滤。"""
        where: list[str] = []
        params: list = []
        if thread_id:
            where.append("thread_id = ?")
            params.append(thread_id)
        if days and days > 0:
            where.append("ts >= ?")
            params.append(int(time.time()) - int(days) * 86400)
        if tenant_id is not None:
            where.append("tenant_id = ?")
            params.append(tenant_id)
        clause = "WHERE " + " AND ".join(where) if where else ""

        with self._lock:
            model_rows = self._conn.execute(
                f"""
                SELECT COALESCE(model_label, ''), COUNT(*),
                       COALESCE(SUM(input_tokens), 0),
                       COALESCE(SUM(output_tokens), 0),
                       COALESCE(SUM(duration_ms), 0),
                       COALESCE(SUM(cached_tokens), 0)
                FROM audit_event
                {clause}
                {"AND" if where else "WHERE"} kind = 'llm_call'
                GROUP BY COALESCE(model_label, '')
                ORDER BY COALESCE(SUM(input_tokens), 0) + COALESCE(SUM(output_tokens), 0) DESC
                """,
                params,
            ).fetchall()
            tool_rows = self._conn.execute(
                f"""
                SELECT COALESCE(tool_name, kind), COUNT(*),
                       COALESCE(SUM(result_size), 0),
                       COALESCE(SUM(duration_ms), 0),
                       COALESCE(SUM(CASE WHEN error IS NULL OR error = '' THEN 0 ELSE 1 END), 0)
                FROM audit_event
                {clause}
                {"AND" if where else "WHERE"} kind IN ('tool_call', 'shell_approve', 'shell_reject', 'mcp_connect', 'model_swap')
                GROUP BY COALESCE(tool_name, kind)
                ORDER BY 2 DESC
                """,
                params,
            ).fetchall()

        return {
            "models": [
                {
                    "model_label": row[0] or None,
                    "calls": int(row[1] or 0),
                    "input_tokens": int(row[2] or 0),
                    "output_tokens": int(row[3] or 0),
                    "duration_ms": int(row[4] or 0),
                    "cached_tokens": int(row[5] or 0),
                }
                for row in model_rows
            ],
            "tools": [
                {
                    "name": row[0] or None,
                    "calls": int(row[1] or 0),
                    "result_size": int(row[2] or 0),
                    "duration_ms": int(row[3] or 0),
                    "errors": int(row[4] or 0),
                }
                for row in tool_rows
            ],
        }

    def import_audit_events(
        self,
        events: list[dict],
        *,
        thread_id: str,
        tenant_id: str | None = None,
    ) -> int:
        """把导出的审计事件恢复到新 thread_id 下，返回写入数量。"""
        count = 0
        for event in events:
            if not isinstance(event, dict):
                continue
            self.log_audit_event(
                thread_id,
                ts=_int_or_none(event.get("ts")) or int(time.time()),
                kind=str(event.get("kind") or "imported"),
                tool_name=event.get("tool_name"),
                args_summary=event.get("args_summary"),
                result_size=_int_or_none(event.get("result_size")),
                model_label=event.get("model_label"),
                input_tokens=_int_or_none(event.get("input_tokens")),
                output_tokens=_int_or_none(event.get("output_tokens")),
                duration_ms=_int_or_none(event.get("duration_ms")),
                intent=event.get("intent"),
                error=event.get("error"),
                tenant_id=tenant_id,
                cached_tokens=_int_or_none(event.get("cached_tokens")),
            )
            count += 1
        return count


def _row_to_audit(row) -> AuditEvent:
    (
        event_id,
        thread_id,
        ts,
        kind,
        tool_name,
        args_summary,
        result_size,
        model_label,
        input_tokens,
        output_tokens,
        duration_ms,
        intent,
        error,
        tenant_id,
        cached_tokens,
    ) = row
    return AuditEvent(
        id=int(event_id or 0),
        thread_id=thread_id,
        ts=int(ts or 0),
        kind=kind,
        tool_name=tool_name,
        args_summary=args_summary,
        result_size=_int_or_none(result_size),
        model_label=model_label,
        input_tokens=_int_or_none(input_tokens),
        output_tokens=_int_or_none(output_tokens),
        duration_ms=_int_or_none(duration_ms),
        intent=intent,
        error=error,
        tenant_id=tenant_id,
        cached_tokens=_int_or_none(cached_tokens),
    )
