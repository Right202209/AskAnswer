"""带耗时计数的终端 spinner —— 用在节点之间的“等待间隙”里。

设计要点：
- 后台线程刷帧；外面主线程照常 print，只要 spinner 在动我们就负责擦掉自己那一行
  （``\\r\\033[K``），避免污染输出；
- 同一个 ``Spinner`` 可以多次 ``transition(text)``：切换文案的同时把秒表归零，
  这样每个节点的耗时是相对独立的；
- 非 TTY 时 ``start`` 直接 no-op：CI/管道场景不该出现转圈字符。
"""
from __future__ import annotations

import sys
import threading
import time

# 用 Braille 的 10 帧动画；视觉上比经典 ``|/-\\`` 圆滑得多
_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ANSI（与 cli.C 重复一份；这里不导入 cli，避免循环依赖）
_ORANGE = "\033[38;5;214m"
_DIM = "\033[2m"
_RESET = "\033[0m"


class Spinner:
    """非阻塞 spinner。

    用法::

        sp = Spinner("Thinking…")
        sp.start()
        ...                   # 主线程做事
        sp.transition("Searching…")  # 节点切换：归零秒表 + 换文案
        ...
        sp.stop()             # 擦掉自己那一行，把光标停在行首

    或者用上下文管理器 ``with Spinner("…") as sp: …``。
    """

    def __init__(self, text: str = "Thinking…", interval: float = 0.08):
        self._text = text
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_at = 0.0
        # 主线程往 stdout 写时要先暂停画面，避免字符交错
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        """启动后台刷帧线程；非 TTY 时静默放弃。"""
        if not _is_tty():
            return
        if self._thread is not None:
            return
        self._stop.clear()
        self._start_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止刷帧并擦掉当前 spinner 行。"""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._erase()
        self._thread = None

    # ── runtime control ──────────────────────────────────────

    def transition(self, text: str) -> None:
        """切到下一阶段：换文案 + 归零秒表（旧阶段的耗时由调用方自己拿走）。"""
        with self._lock:
            self._text = text
            self._start_at = time.monotonic()

    def elapsed(self) -> float:
        """从最近一次 ``start``/``transition`` 到现在的秒数。"""
        return time.monotonic() - self._start_at

    def freeze_for(self, fn) -> None:
        """暂停 spinner，执行 ``fn``（通常是一次 print），然后立即恢复。

        用途：节点完成时主线程要打 ``⏺ Marker`` —— 我们不希望和 spinner 抢同一行。
        """
        if self._thread is None:
            fn()
            return
        with self._lock:
            self._erase()
            fn()

    # ── internals ────────────────────────────────────────────

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            with self._lock:
                elapsed = time.monotonic() - self._start_at
                frame = _FRAMES[i % len(_FRAMES)]
                line = (
                    f"  {_ORANGE}{frame}{_RESET} "
                    f"{_DIM}{self._text}{_RESET} "
                    f"{_DIM}({elapsed:.1f}s){_RESET}"
                )
                # \r 回行首；\033[K 擦到行尾，避免上一帧残留
                sys.stdout.write(f"\r{line}\033[K")
                sys.stdout.flush()
            time.sleep(self._interval)
            i += 1

    @staticmethod
    def _erase() -> None:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


__all__ = ["Spinner"]
