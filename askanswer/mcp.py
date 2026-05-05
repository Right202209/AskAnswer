# MCP（Model Context Protocol）客户端管理器。
#
# 关键设计：所有 MCP 调用都被 hop 到一个独立后台线程上的 asyncio 事件循环里执行。
# 这样每个 server 的 async context manager 进出都发生在同一个 task 上，避免
# anyio 在跨任务退出时报 cancel-scope 错误。
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


# 安全的服务名字符集合 —— 用于把 URL 派生出可读、可作为前缀的服务名
_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass
class _ServerEntry:
    """单个 MCP server 的运行时记录。"""
    name: str
    url: str | None
    transport: str            # "streamable_http" | "sse" | "stdio"
    session: ClientSession    # 与 server 建立的会话
    tools: list[dict[str, Any]]  # 启动时拉取到的工具列表
    _close_event: asyncio.Event  # 通知后台 task 退出 ctx manager 的事件
    _task: asyncio.Task = field(repr=False)  # 后台 runner task


class MCPClientManager:
    """管理一到多个 MCP 客户端会话。

    支持三种传输：
      - ``streamable_http``：通过 ``https://host/mcp`` 这类 URL 连接（首选）
      - ``sse``：URL 形如 ``https://host/sse`` 时自动选用
      - ``stdio``：启动子进程作为 MCP server

    聚合多个 server 的工具时，工具名以 ``<server>__<tool>`` 暴露，
    调用 :meth:`call_tool` 时按前缀路由。
    """

    # 工具名前缀与 server 名之间的分隔符
    _PREFIX_SEP = "__"

    def __init__(self) -> None:
        # 后台事件循环 + 守护线程，所有 MCP 协程都在这里跑
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
        """连接一个 HTTP(S) MCP 服务，返回最终注册的服务名。"""
        self._ensure_open()
        # 没指定名字时从 URL 派生（host + path 拼成的安全字符串）
        resolved = (name or self._name_from_url(url)).strip()
        if not resolved:
            raise ValueError("无法从 URL 推导服务名，请显式指定 name")
        # 没指定传输时按 URL 路径自动判别（/sse 走 SSE，否则 streamable_http）
        transport = (transport or self._guess_transport(url)).lower()
        if transport not in {"streamable_http", "sse"}:
            raise ValueError(f"不支持的 URL 传输：{transport}")
        # 提前占名，避免并发场景下连成功后才发现重名
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
            # 启动失败要记得释放占名
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
        """以子进程方式启动一个 stdio MCP server。"""
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
        """:meth:`add_stdio` 的向后兼容别名。"""
        self.add_stdio(name=name, command=command, args=args, env=env)

    def remove(self, name: str) -> bool:
        """断开并移除指定名称的 MCP 服务，返回是否真的找到并移除。"""
        with self._lock:
            entry = self._servers.pop(name, None)
        if entry is None:
            return False
        try:
            # 关闭时即使报错也忽略，避免影响其它服务的清理
            self._submit(self._close_entry(entry))
        except Exception:
            pass
        return True

    def list_servers(self) -> list[dict[str, Any]]:
        """返回所有已连接 server 的概要信息（不含工具详情）。"""
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
        """跨所有 server 聚合工具列表。

        每条返回项包含：完整命名（``server__tool``）、所属 server、原始 schema。
        """
        combined: list[dict[str, Any]] = []
        with self._lock:
            entries = list(self._servers.values())
        for entry in entries:
            # 过滤指定 server
            if server is not None and entry.name != server:
                continue
            for t in entry.tools:
                combined.append(
                    {
                        # 关键：工具名前缀 server 名，避免不同 server 同名冲突
                        "name": f"{entry.name}{self._PREFIX_SEP}{t['name']}",
                        "server": entry.name,
                        "original_name": t["name"],
                        "description": t.get("description") or "",
                        "input_schema": t.get("input_schema") or {},
                    }
                )
        return combined

    def call_tool(self, name: str, args: dict[str, Any] | None = None) -> str:
        """按 ``server__tool`` 完整名调用工具，返回拼接后的字符串结果。"""
        server_name, tool_name = self._split_tool_name(name)
        with self._lock:
            entry = self._servers.get(server_name)
        if entry is None:
            raise RuntimeError(f"未找到 MCP 服务：{server_name}")
        # 真正的网络/IPC 调用 hop 到后台 loop 上执行
        result = self._submit(
            self._call_tool_async(entry=entry, tool_name=tool_name, args=args or {})
        )
        return self._stringify_result(result)

    def has_server(self, name: str) -> bool:
        with self._lock:
            return name in self._servers

    def close(self) -> None:
        """优雅关闭所有 server 与后台事件循环；可重入。"""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            entries = list(self._servers.values())
            self._servers.clear()
        # 逐个通知 server 退出 ctx manager
        for entry in entries:
            try:
                self._submit(self._close_entry(entry))
            except Exception:
                pass
        # 停掉事件循环、等线程退出，最后关闭 loop
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
        """提前占用名字，防止并发添加同名 server。"""
        with self._lock:
            if name in self._servers:
                raise RuntimeError(f"已存在同名 MCP 服务：{name}")

    def _release_name(self, name: str) -> None:
        with self._lock:
            self._servers.pop(name, None)

    @classmethod
    def _split_tool_name(cls, name: str) -> tuple[str, str]:
        """把 ``server__tool`` 分回 (server, tool)。"""
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
        """根据 URL 路径自动猜测传输类型。"""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"MCP URL 需为 http(s)：{url}")
        path = (parsed.path or "").rstrip("/").lower()
        # /sse 结尾通常是 SSE 端点，其它默认走 streamable_http
        if path.endswith("/sse"):
            return "sse"
        return "streamable_http"

    @classmethod
    def _name_from_url(cls, url: str) -> str:
        """从 URL 派生一个可读、可作为工具前缀的服务名。"""
        parsed = urlparse(url)
        host = parsed.hostname or "mcp"
        path = (parsed.path or "").strip("/")
        raw = f"{host}-{path}" if path else host
        # 用 _NAME_SAFE 把非法字符替换成 -，并去掉首尾多余连字符
        cleaned = _NAME_SAFE.sub("-", raw).strip("-")
        return cleaned or host

    @staticmethod
    def _stringify_result(result: Any) -> str:
        """把 MCP 工具调用结果统一拼成字符串。

        result.content 通常是一组带 text 字段的对象，把它们拼起来即可。
        """
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
        """HTTP/SSE 类传输的会话上下文。yield (session, tools) 后，
        在外层 close_event 被设置之前一直保持连接。"""
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
                # streams 是 (read, write[, ...]) 元组，前两个是双向流
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
        else:  # pragma: no cover - 调用前已校验
            raise ValueError(f"不支持的 MCP 传输：{transport}")

    @asynccontextmanager
    async def _stdio_cm(
        self,
        *,
        command: str,
        args: list[str],
        env: dict[str, str] | None,
    ) -> AsyncIterator[tuple[ClientSession, list[dict[str, Any]]]]:
        """stdio 子进程方式的会话上下文。"""
        params = StdioServerParameters(command=command, args=args, env=env)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await self._collect_tools(session)
                yield session, tools

    @staticmethod
    async def _collect_tools(session: ClientSession) -> list[dict[str, Any]]:
        """会话建立后第一时间拉取工具列表并整理成普通字典。"""
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
        """启动一个 server：用 ready future 等待握手成功后返回 _ServerEntry。"""
        ready: asyncio.Future[tuple[ClientSession, list[dict[str, Any]]]] = (
            self._loop.create_future()
        )
        close_event = asyncio.Event()

        async def _runner() -> None:
            # 在同一个 task 里完成 ctx manager 的 enter/exit，规避 cancel-scope 限制
            try:
                async with cm_factory() as (session, tools):
                    if not ready.done():
                        ready.set_result((session, tools))
                    # 一直等到外部 close() 触发 close_event 才退出 ctx
                    await close_event.wait()
            except Exception as exc:
                if not ready.done():
                    ready.set_exception(exc)

        task = asyncio.ensure_future(_runner())
        try:
            session, tools = await ready
        except Exception:
            # 启动失败：尽量优雅地把 task 收尾，再把异常向上抛
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
        # 单纯转发到 session.call_tool；放在 async 函数里以便 _submit 调度
        return await entry.session.call_tool(tool_name, args)

    async def _close_entry(self, entry: _ServerEntry) -> None:
        """请求 server runner 退出 ctx manager；超时则强制取消。"""
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
        """后台线程主循环：把 self._loop 设成本线程的事件循环并跑起来。"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """把协程从主线程提交到后台 loop，并阻塞等待结果。"""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


# ── Module-level singleton helpers ────────────────────────────────

# 单例：整个进程共享一个 MCP 管理器；CLI 退出时由 atexit 关闭
_manager: MCPClientManager | None = None
_manager_lock = Lock()


def get_manager() -> MCPClientManager:
    """获取（懒初始化）模块级 MCP 管理器。"""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = MCPClientManager()
        return _manager


def shutdown_manager() -> None:
    """关闭模块级管理器；CLI 通过 atexit 调用。"""
    global _manager
    with _manager_lock:
        mgr, _manager = _manager, None
    if mgr is not None:
        mgr.close()
