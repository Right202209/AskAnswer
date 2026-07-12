# AskAnswer 命令行入口包。
#
# 由 `cli.py`（2510 行单文件）拆分而来（计划 C1）。模块地图：
# - theme / text            视觉主题与文本度量原语
# - render / progress       终端渲染与节点进度标记
# - confirm                 HITL 确认菜单
# - stream                  问答主流程（消费 runner 的事件流）
# - repl                    交互式 REPL 循环 + `!<cmd>`
# - app                     启动装配（参数、图导出、MCP 自动重连、telemetry）
# - commands/               斜杠命令按域拆分（model/mcp/threads/timetravel/audit/transfer）
#
# ``main`` 是唯一对外入口（`askanswer` 脚本与 `python -m askanswer` 都走它）；
# 其余符号做向后兼容的再导出，历史上从 ``askanswer.cli`` 直接 import 的名字不受影响。
from __future__ import annotations

import atexit
import uuid

from ..graph import create_search_assistant
from ..mcp import shutdown_manager as _mcp_shutdown
from ..persistence import shutdown_persistence
from ..registry import get_registry
from .app import (
    _autoconnect_mcp_profile,
    _init_telemetry,
    build_parser,
    export_graph,
)
from .commands import handle_command
from .render import render_answer, render_error, status_block, welcome_box
from .repl import interactive_loop, run_bang_command
from .stream import stream_query

__all__ = [
    "main",
    "stream_query",
    "interactive_loop",
    "handle_command",
    "run_bang_command",
    "render_answer",
    "render_error",
    "status_block",
    "welcome_box",
    "build_parser",
    "export_graph",
]


def main() -> int:
    """CLI 程序入口：解析参数 → 注册 atexit → 进入对应模式。"""
    parser = build_parser()
    args = parser.parse_args()

    # 保证退出时关闭 MCP 后台 loop，避免守护线程残留
    atexit.register(_mcp_shutdown)
    # 关闭 SQLite 连接 —— atexit 是 LIFO，shutdown_persistence 会先于 _mcp_shutdown 调用，
    # 两者无依赖关系，顺序无所谓。
    atexit.register(shutdown_persistence)

    # 仅导图模式：不需要构建主图，直接输出 Mermaid（draw_search_assistant_mermaid
    # 内部走 InMemorySaver，不会触发 ~/.askanswer/state.db 创建）
    if args.graph is not None:
        return export_graph(args.graph)

    try:
        # seed 一次注册表（内置 + sql + 任何已存在的 MCP），后续节点直接用
        get_registry()
        # 按环境变量装配可观测性 exporter（LangSmith / OTEL）；未开启则零开销
        _init_telemetry()
        # 从 profile 自动重连上次连过的 MCP server（失败不阻塞启动）
        _autoconnect_mcp_profile()
        app = create_search_assistant()
    except Exception as exc:
        render_error(f"初始化失败: {exc}")
        return 1

    # 单次问答模式：命令行直接给了 question
    if args.question:
        thread_id = str(uuid.uuid4())
        try:
            stream_query(app, args.question, thread_id)
        except Exception as exc:
            render_error(str(exc))
            return 1
        return 0

    # 没问题参数则进入 REPL
    return interactive_loop(app)


if __name__ == "__main__":
    raise SystemExit(main())
