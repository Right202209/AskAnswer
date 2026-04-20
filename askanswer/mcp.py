from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Any
from urllib.parse import urlparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass
class _ServerEntry:
    name: str
    url: str | None
    transport: str            # "streamable_http" | "sse" | "stdio"
    session: ClientSession
    tools: list[dict[str, Any]]
    _close_event: asyncio.Event
    _task: asyncio.Task = field(repr=False)


class MCPClientManager:
    """Manage one or more MCP client sessions.

    支持三种传输：
      - ``streamable_http``：通过 ``https://host/mcp`` 这类 URL 连接（首选）
      - ``sse``：URL 形如 ``https://host/sse`` 时自动选用
      - ``stdio``：启动子进程作为 MCP server

    聚合多个 server 的工具时，工具名以 ``<server>__<tool>`` 暴露，
    调用 :meth:`call_tool` 时按前缀路由。
    """

    _PREFIX_SEP = "__"

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_loop, name="mcp-loop", daemon=True)
        self._thread.start()
        self._servers: dict[str, _ServerEntry] = {}
        self._lock = Lock()
        self._closed = False

    # ── Public API ───────────────────────────────────────────────────

    def add_url(
        self,
        url: str,
        *,
        name: str | None = None,
        transport: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        """Connect to an HTTP(S) MCP server. Returns the registered name."""
        self._ensure_open()
        resolved = (name or self._name_from_url(url)).strip()
        if not resolved:
            raise ValueError("无法从 URL 推导服务名，请显式指定 name")
        transport = (transport or self._guess_transport(url)).lower()
        if transport not in {"streamable_http", "sse"}:
            raise ValueError(f"不支持的 URL 传输：{transport}")
        self._reserve_name(resolved)
        try:
            entry = self._submit(
                self._start_server(
                    name=resolved,
                    url=url,
                    transport=transport,
                    cm_factory=lambda: self._url_cm(
                        url=url, transport=transport, headers=headers
                    ),
                )
            )
        except Exception:
            self._release_name(resolved)
            raise
        with self._lock:
            self._servers[resolved] = entry
        return resolved

    def add_stdio(
        self,
        *,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        self._ensure_open()
        resolved = (name or "").strip()
        if not resolved:
            raise ValueError("stdio MCP 服务必须提供 name")
        self._reserve_name(resolved)
        try:
            entry = self._submit(
                self._start_server(
                    name=resolved,
                    url=None,
                    transport="stdio",
                    cm_factory=lambda: self._stdio_cm(
                        command=command, args=args or [], env=env
                    ),
                )
            )
        except Exception:
            self._release_name(resolved)
            raise
        with self._lock:
            self._servers[resolved] = entry
        return resolved

    def connect(
        self,
        *,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Backwards-compatible alias for :meth:`add_stdio`."""
        self.add_stdio(name=name, command=command, args=args, env=env)

    def remove(self, name: str) -> bool:
        with self._lock:
            entry = self._servers.pop(name, None)
        if entry is None:
            return False
        try:
            self._submit(self._close_entry(entry))
        except Exception:
            pass
        return True

    def list_servers(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": e.name,
                    "url": e.url,
                    "transport": e.transport,
                    "tools": len(e.tools),
                }
                for e in self._servers.values()
            ]

    def list_tools(self, *, server: str | None = None) -> list[dict[str, Any]]:
        """Aggregated tool list across servers.

        Each returned item has the fully-qualified ``name`` (``server__tool``),
        the ``server`` it belongs to, and the original ``input_schema``.
        """
        combined: list[dict[str, Any]] = []
        with self._lock:
            entries = list(self._servers.values())
        for entry in entries:
            if server is not None and entry.name != server:
                continue
            for t in entry.tools:
                combined.append(
                    {
                        "name": f"{entry.name}{self._PREFIX_SEP}{t['name']}",
                        "server": entry.name,
                        "original_name": t["name"],
                        "description": t.get("description") or "",
                        "input_schema": t.get("input_schema") or {},
                    }
                )
        return combined

    def call_tool(self, name: str, args: dict[str, Any] | None = None) -> str:
        server_name, tool_name = self._split_tool_name(name)
        with self._lock:
            entry = self._servers.get(server_name)
        if entry is None:
            raise RuntimeError(f"未找到 MCP 服务：{server_name}")
        result = self._submit(
            self._call_tool_async(entry=entry, tool_name=tool_name, args=args or {})
        )
        return self._stringify_result(result)

    def has_server(self, name: str) -> bool:
        with self._lock:
            return name in self._servers

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            entries = list(self._servers.values())
            self._servers.clear()
        for entry in entries:
            try:
                self._submit(self._close_entry(entry))
            except Exception:
                pass
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if not self._loop.is_closed():
            self._loop.close()

    # ── Private: naming / transports ─────────────────────────────────

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MCP 管理器已关闭")

    def _reserve_name(self, name: str) -> None:
        with self._lock:
            if name in self._servers:
                raise RuntimeError(f"已存在同名 MCP 服务：{name}")

    def _release_name(self, name: str) -> None:
        with self._lock:
            self._servers.pop(name, None)

    @classmethod
    def _split_tool_name(cls, name: str) -> tuple[str, str]:
        if cls._PREFIX_SEP not in name:
            raise ValueError(
                f"工具名应形如 '<server>{cls._PREFIX_SEP}<tool>'，收到：{name}"
            )
        server, _, tool = name.partition(cls._PREFIX_SEP)
        if not server or not tool:
            raise ValueError(f"工具名格式非法：{name}")
        return server, tool

    @staticmethod
    def _guess_transport(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"MCP URL 需为 http(s)：{url}")
        path = (parsed.path or "").rstrip("/").lower()
        if path.endswith("/sse"):
            return "sse"
        return "streamable_http"

    @classmethod
    def _name_from_url(cls, url: str) -> str:
        parsed = urlparse(url)
        host = parsed.hostname or "mcp"
        path = (parsed.path or "").strip("/")
        raw = f"{host}-{path}" if path else host
        cleaned = _NAME_SAFE.sub("-", raw).strip("-")
        return cleaned or host

    @staticmethod
    def _stringify_result(result: Any) -> str:
        content = getattr(result, "content", None) or []
        chunks: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                chunks.append(str(text))
                continue
            if isinstance(item, dict) and item.get("text"):
                chunks.append(str(item["text"]))
                continue
            chunks.append(str(item))
        return "\n".join(chunks).strip() or str(result)

    # ── Private: async lifecycle ─────────────────────────────────────

    @asynccontextmanager
    async def _url_cm(
        self,
        *,
        url: str,
        transport: str,
        headers: dict[str, str] | None,
    ) -> AsyncIterator[tuple[ClientSession, list[dict[str, Any]]]]:
        if transport == "streamable_http":
            try:
                from mcp.client.streamable_http import streamablehttp_client
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "当前 mcp SDK 不支持 streamable_http，请升级 mcp 或使用 SSE 端点"
                ) from exc
            kwargs: dict[str, Any] = {}
            if headers:
                kwargs["headers"] = headers
            async with streamablehttp_client(url, **kwargs) as streams:
                read_stream, write_stream = streams[0], streams[1]
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await self._collect_tools(session)
                    yield session, tools
        elif transport == "sse":
            from mcp.client.sse import sse_client

            kwargs = {"headers": headers} if headers else {}
            async with sse_client(url, **kwargs) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await self._collect_tools(session)
                    yield session, tools
        else:  # pragma: no cover - guarded at caller
            raise ValueError(f"不支持的 MCP 传输：{transport}")

    @asynccontextmanager
    async def _stdio_cm(
        self,
        *,
        command: str,
        args: list[str],
        env: dict[str, str] | None,
    ) -> AsyncIterator[tuple[ClientSession, list[dict[str, Any]]]]:
        params = StdioServerParameters(command=command, args=args, env=env)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await self._collect_tools(session)
                yield session, tools

    @staticmethod
    async def _collect_tools(session: ClientSession) -> list[dict[str, Any]]:
        resp = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in resp.tools
        ]

    async def _start_server(
        self,
        *,
        name: str,
        url: str | None,
        transport: str,
        cm_factory,
    ) -> _ServerEntry:
        ready: asyncio.Future[tuple[ClientSession, list[dict[str, Any]]]] = (
            self._loop.create_future()
        )
        close_event = asyncio.Event()

        async def _runner() -> None:
            try:
                async with cm_factory() as (session, tools):
                    if not ready.done():
                        ready.set_result((session, tools))
                    await close_event.wait()
            except Exception as exc:
                if not ready.done():
                    ready.set_exception(exc)

        task = asyncio.ensure_future(_runner())
        try:
            session, tools = await ready
        except Exception:
            close_event.set()
            task.cancel()
            try:
                await task
            except Exception:
                pass
            raise

        return _ServerEntry(
            name=name,
            url=url,
            transport=transport,
            session=session,
            tools=tools,
            _close_event=close_event,
            _task=task,
        )

    async def _call_tool_async(
        self,
        *,
        entry: _ServerEntry,
        tool_name: str,
        args: dict[str, Any],
    ) -> Any:
        return await entry.session.call_tool(tool_name, args)

    async def _close_entry(self, entry: _ServerEntry) -> None:
        entry._close_event.set()
        try:
            await asyncio.wait_for(entry._task, timeout=5.0)
        except asyncio.TimeoutError:
            entry._task.cancel()
            try:
                await entry._task
            except Exception:
                pass
        except Exception:
            pass

    # ── Loop plumbing ────────────────────────────────────────────────

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro: Coroutine[Any, Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


# ── Module-level singleton helpers ────────────────────────────────

_manager: MCPClientManager | None = None
_manager_lock = Lock()


def get_manager() -> MCPClientManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = MCPClientManager()
        return _manager


def shutdown_manager() -> None:
    global _manager
    with _manager_lock:
        mgr, _manager = _manager, None
    if mgr is not None:
        mgr.close()
