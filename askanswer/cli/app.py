# CLI 启动装配：参数解析、Mermaid 图导出、MCP profile 自动重连、telemetry 初始化。
from __future__ import annotations

import argparse
from pathlib import Path

from .. import mcp_profile
from ..graph import draw_search_assistant_mermaid
from ..mcp import get_manager as _mcp_manager
from ..registry import get_registry
from .render import render_error
from .theme import C, _console


def build_parser() -> argparse.ArgumentParser:
    """命令行参数解析器：支持单次问答、--graph 导出图。"""
    parser = argparse.ArgumentParser(description="AskAnswer 命令行工具")
    parser.add_argument(
        "--graph",
        nargs="?",
        const="-",
        metavar="PATH",
        help="生成 LangGraph Mermaid 图；不填 PATH 时输出到终端",
    )
    parser.add_argument("question", nargs="?", help="要提问的内容")
    return parser


def export_graph(target: str) -> int:
    """导出 LangGraph Mermaid 图到指定路径或 stdout。"""
    try:
        mermaid = draw_search_assistant_mermaid()
    except Exception as exc:
        render_error(f"生成图失败: {exc}")
        return 1

    # "-" 表示输出到 stdout
    if target == "-":
        print(mermaid)
        return 0

    # 否则写入文件，目录不存在就先创建
    path = Path(target)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mermaid, encoding="utf-8")
    except Exception as exc:
        render_error(f"写入图文件失败: {exc}")
        return 1

    print(f"\n  {C.GREEN}✓ LangGraph 图已生成:{C.RESET} {path}\n")
    return 0


def _init_telemetry() -> None:
    """按环境变量装配可观测性 exporter；lazy import 避免顶层依赖 telemetry。"""
    try:
        from .. import telemetry

        telemetry.init_telemetry()
    except Exception as exc:
        # 可观测性初始化失败绝不能阻断 CLI 启动。
        _console.print(f"  [warning]⚠ telemetry 初始化失败：{exc}[/]")


def _autoconnect_mcp_profile() -> None:
    """启动时从 ``~/.askanswer/mcp.json`` 逐项重连 MCP server。

    单条失败仅告警、不阻塞启动（server 可能已下线）。全部尝试完再 refresh 一次
    注册表，让本次成功连上的工具进入工具表。
    """
    try:
        records = mcp_profile.load()
    except Exception as exc:
        _console.print(f"  [warning]⚠ 读取 MCP profile 失败：{exc}[/]")
        return
    if not records:
        return
    connected = 0
    for record in records:
        try:
            _reconnect_mcp_record(record)
            connected += 1
        except Exception as exc:
            name = record.get("name") or "(unknown)"
            _console.print(f"  [warning]⚠ MCP 自动重连失败 {name}：{exc}[/]")
    if connected:
        get_registry().refresh_mcp()
        _console.print(f"  [subtle]已从 profile 自动重连 {connected} 个 MCP 服务[/]")


def _reconnect_mcp_record(record: dict) -> None:
    """按 profile 记录里的 transport 选择重连方式。"""
    transport = str(record.get("transport") or "").lower()
    name = record.get("name")
    if transport == "stdio":
        _mcp_manager().add_stdio(
            name=name,
            command=record.get("command") or "",
            args=list(record.get("args") or []),
            env=record.get("env"),
        )
    else:
        _mcp_manager().add_url(
            record.get("url") or "",
            name=name,
            transport=transport or None,
            headers=record.get("headers"),
        )
