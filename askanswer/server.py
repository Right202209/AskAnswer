# AskAnswer HTTP/SSE server —— 标准库实现，零新增依赖。
#
# 端点：GET /health · GET /v1/interrupt?thread_id=… · POST /v1/query · POST /v1/resume
# （请求/事件契约详见 docs/important-documentation-c3-http-sse-server.md）
#
# 与 CLI 消费同一事件序列（askanswer.runner）、同一图与 checkpointer：确认类工具经
# HTTP 时 /query 流在 interrupt 事件后以 done.status=interrupted 结束（挂起态由
# checkpointer 持久化），客户端任意时刻 POST /v1/resume 续跑（decision 形态与 CLI
# 的 Command(resume=...) 值一致）。
#
# 安全基线（本地开发服务定位）：默认只绑 127.0.0.1；可选 ASKANSWER_SERVER_TOKEN 启用
# Bearer 鉴权（仅从 env 读取）；拒绝非 localhost 跨源 + 强制 application/json（CSRF
# 双闸）；请求体/查询长度上限；全局并发与同 thread 并发受限；错误响应不回显内部细节。
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import runner
from .wire import (
    LOCAL_HOSTS,
    RequestError,
    event_wire,
    is_local_origin,
    json_safe,
    normalize_thread_id,
    split_path,
    sse_frame,
)

_LOG = logging.getLogger("askanswer.server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 64 * 1024
MAX_QUERY_CHARS = 8000
# SqliteSaver 全进程共用一条连接（内部有锁），并发跑图收益有限；小上限即是限流。
MAX_CONCURRENT_RUNS = 2
# 套接字读写超时（防 Slowloris / 死客户端占线程）；纯计算间隙不受此限制。
SOCKET_TIMEOUT_SECONDS = 60
TOKEN_ENV = "ASKANSWER_SERVER_TOKEN"

_RUN_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)
_BUSY_THREADS: set[str] = set()
_BUSY_LOCK = threading.Lock()
_APP = None
_APP_LOCK = threading.Lock()


def _get_app():
    """懒加载并缓存编译后的图（多请求线程共享；persistence 用锁保护单连接）。"""
    global _APP
    with _APP_LOCK:
        if _APP is None:
            from .graph import create_search_assistant

            _APP = create_search_assistant()
        return _APP


def _claim_thread(thread_id: str) -> bool:
    with _BUSY_LOCK:
        if thread_id in _BUSY_THREADS:
            return False
        _BUSY_THREADS.add(thread_id)
        return True


def _release_thread(thread_id: str) -> None:
    with _BUSY_LOCK:
        _BUSY_THREADS.discard(thread_id)


class AskAnswerHandler(BaseHTTPRequestHandler):
    server_version = "AskAnswer"
    protocol_version = "HTTP/1.1"
    timeout = SOCKET_TIMEOUT_SECONDS

    def do_GET(self) -> None:  # noqa: N802 —— http.server 固定命名
        try:
            path, params = split_path(self.path)
            if path == "/health":
                self._send_json(HTTPStatus.OK, {"ok": True})
                return
            self._guard()
            if path == "/v1/interrupt":
                self._handle_interrupt_query(params)
                return
            raise RequestError(HTTPStatus.NOT_FOUND, "unknown path")
        except RequestError as exc:
            self._send_error(exc.status, str(exc))
        except Exception:
            self._fail_safely()

    def do_POST(self) -> None:  # noqa: N802
        try:
            path, _ = split_path(self.path)
            self._guard()
            if path == "/v1/query":
                self._handle_query()
            elif path == "/v1/resume":
                self._handle_resume()
            else:
                raise RequestError(HTTPStatus.NOT_FOUND, "unknown path")
        except RequestError as exc:
            self._send_error(exc.status, str(exc))
        except Exception:
            self._fail_safely()

    # ---- 守门：鉴权 / 跨源 / 请求体 ----------------------------------------
    def _guard(self) -> None:
        origin = self.headers.get("Origin")
        if origin and not is_local_origin(origin):
            raise RequestError(HTTPStatus.FORBIDDEN, "cross-origin requests are not allowed")
        expected = os.getenv(TOKEN_ENV) or ""
        if not expected:
            return
        provided = self.headers.get("Authorization") or ""
        if not hmac.compare_digest(provided, f"Bearer {expected}"):
            raise RequestError(HTTPStatus.UNAUTHORIZED, "missing or invalid bearer token")

    def _read_json(self) -> dict:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            raise RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required") from None
        if length <= 0 or length > MAX_BODY_BYTES:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body must be 1..{MAX_BODY_BYTES} bytes")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RequestError(HTTPStatus.BAD_REQUEST, "body must be valid JSON") from None
        if not isinstance(data, dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
        return data

    # ---- 业务端点 ----------------------------------------------------------
    def _handle_query(self) -> None:
        body = self._read_json()
        query = str(body.get("query") or "").strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"query must be 1..{MAX_QUERY_CHARS} chars")
        thread_id = normalize_thread_id(body.get("thread_id")) or uuid.uuid4().hex
        self._stream_leg(thread_id, runner.query_input(query), preview=runner.preview_of(query))

    def _handle_resume(self) -> None:
        body = self._read_json()
        thread_id = normalize_thread_id(body.get("thread_id"))
        if not thread_id:
            raise RequestError(HTTPStatus.BAD_REQUEST, "thread_id is required")
        if "decision" not in body:
            raise RequestError(HTTPStatus.BAD_REQUEST, "decision is required")
        if runner.pending_interrupt(_get_app(), runner.thread_config(thread_id)) is None:
            raise RequestError(HTTPStatus.CONFLICT, "no pending interrupt on this thread")
        self._stream_leg(thread_id, runner.resume_input(body["decision"]), preview=None)

    def _handle_interrupt_query(self, params: dict) -> None:
        thread_id = normalize_thread_id((params.get("thread_id") or [""])[0])
        if not thread_id:
            raise RequestError(HTTPStatus.BAD_REQUEST, "thread_id is required")
        payload = runner.pending_interrupt(_get_app(), runner.thread_config(thread_id))
        self._send_json(HTTPStatus.OK, {"thread_id": thread_id, "interrupt": json_safe(payload)})

    # ---- SSE 泵 ------------------------------------------------------------
    def _stream_leg(self, thread_id: str, graph_input, *, preview: str | None) -> None:
        if not _RUN_SLOTS.acquire(blocking=False):
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "server is at max concurrent runs")
        try:
            if not _claim_thread(thread_id):
                raise RequestError(HTTPStatus.CONFLICT, "thread is busy with another run")
            try:
                self._pump_events(thread_id, graph_input, preview=preview)
            finally:
                _release_thread(thread_id)
        finally:
            _RUN_SLOTS.release()

    def _pump_events(self, thread_id: str, graph_input, *, preview: str | None) -> None:
        events = runner.run_leg(
            _get_app(), graph_input,
            thread_id=thread_id, context=runner.runtime_context_from_env(), preview=preview,
        )
        self._begin_sse(thread_id)
        status = "completed"
        try:
            status = self._forward_events(events)
        except (BrokenPipeError, ConnectionResetError):
            _LOG.info("client disconnected: thread=%s", thread_id)
            return
        except Exception:
            _LOG.exception("run failed: thread=%s", thread_id)
            self._write_frame_quietly("error", {"message": "internal error"})
            status = "failed"
        finally:
            events.close()  # 提前退出也触发 runner 的记账 finally
        self._write_frame_quietly("done", {"status": status, "thread_id": thread_id})

    def _forward_events(self, events) -> str:
        status = "completed"
        for event in events:
            name, data = event_wire(event)
            self._write_frame(name, data)
            if event.kind == runner.EVENT_INTERRUPT:
                status = "interrupted"
        return status

    def _begin_sse(self, thread_id: str) -> None:
        self._sse_started = True
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self._write_frame("meta", {"thread_id": thread_id})

    def _write_frame(self, name: str, data) -> None:
        self.wfile.write(sse_frame(name, data))
        self.wfile.flush()

    def _write_frame_quietly(self, name: str, data) -> None:
        try:
            self._write_frame(name, data)
        except OSError:
            pass

    # ---- 响应与日志 ---------------------------------------------------------
    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"code": int(status), "message": message}})

    def _fail_safely(self) -> None:
        """未预期异常的收口：栈只进服务端日志；SSE 已开流时不再发 HTTP 错误响应。"""
        _LOG.exception("unhandled server error")
        if getattr(self, "_sse_started", False):
            return
        self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")

    def log_message(self, format: str, *args) -> None:  # noqa: A002 —— 基类签名
        _LOG.info("%s %s", self.address_string(), format % args)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="askanswer-server", description="AskAnswer HTTP/SSE server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"绑定地址（默认 {DEFAULT_HOST}，仅本机）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口（默认 {DEFAULT_PORT}）")
    args = parser.parse_args(argv)
    if args.host not in LOCAL_HOSTS and not os.getenv(TOKEN_ENV):
        _LOG.warning("绑定非本机地址 %s 且未设置 %s：任何能访问该端口的人都可调用图。", args.host, TOKEN_ENV)
    server = ThreadingHTTPServer((args.host, args.port), AskAnswerHandler)
    server.daemon_threads = True
    _LOG.info("AskAnswer server listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _LOG.info("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
