from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from threading import Thread
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClientManager:

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._stdio_cm: Any | None = None
        self._session_cm: Any | None = None
        self._session: ClientSession | None = None
        self._connected = False
        self._server_name: str | None = None

    def connect(
        self,
        *,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if self._connected:
            if self._server_name == name:
                return
            raise RuntimeError(f"已连接 MCP 服务：{self._server_name}")

        self._submit(
            self._connect_async(
                name=name,
                command=command,
                args=args or [],
                env=env,
            )
        )

    def list_tools(self) -> list[dict[str, Any]]:
        response = self._submit(self._list_tools_async())
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

    def call_tool(self, name: str, args: dict[str, Any] | None = None) -> str:
        result = self._submit(self._call_tool_async(name=name, args=args or {}))
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

    def close(self) -> None:
        if self._connected:
            self._submit(self._close_async())

        if self._loop.is_running():
            self._stop_loop()
        if self._thread.is_alive():
            self._thread.join()
        if not self._loop.is_closed():
            self._loop.close()

    async def _connect_async(
        self,
        *,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None,
    ) -> None:
        server_params = StdioServerParameters(command=command, args=args, env=env)
        stdio_cm = stdio_client(server_params)
        session_cm: Any | None = None

        try:
            read_stream, write_stream = await stdio_cm.__aenter__()
            session_cm = ClientSession(read_stream, write_stream)
            session = await session_cm.__aenter__()
            await session.initialize()
        except Exception:
            if session_cm is not None:
                await session_cm.__aexit__(None, None, None)
            await stdio_cm.__aexit__(None, None, None)
            raise

        self._stdio_cm = stdio_cm
        self._session_cm = session_cm
        self._session = session
        self._connected = True
        self._server_name = name

    async def _list_tools_async(self) -> Any:
        session = self._require_session()
        return await session.list_tools()

    async def _call_tool_async(self, *, name: str, args: dict[str, Any]) -> Any:
        session = self._require_session()
        return await session.call_tool(name, args)

    async def _close_async(self) -> None:
        session_cm = self._session_cm
        stdio_cm = self._stdio_cm
        self._session = None
        self._session_cm = None
        self._stdio_cm = None
        self._connected = False
        self._server_name = None

        if session_cm is not None:
            await session_cm.__aexit__(None, None, None)
        if stdio_cm is not None:
            await stdio_cm.__aexit__(None, None, None)

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("MCP 尚未连接")
        return self._session

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _stop_loop(self) -> None:
        if self._loop.is_running():
            self._loop.stop()

    def _submit(self, coro: Coroutine[Any, Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()