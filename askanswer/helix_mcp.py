"""把 Helix 规格演化子图作为 MCP server 暴露给外部 agent 调用。

用法（作为一个 stdio MCP server 启动）::

    python -m askanswer.helix_mcp

外部 MCP client（含 AskAnswer 自身的 ``/mcp add_stdio``）即可 list/call 到
``helix_spec_loop`` 工具。

不变量（对齐 execution-plan Phase 3.4）：
- **禁止顶层 import ``graph.py``**：只 reuse ``helix.agent.run_helix_agent``，因此不会
  触发主图编译或 ``get_persistence()`` 初始化（Helix 子图自身不带 checkpointer）。
- **非 TTY 回退 default_answer**：stdio server 环境下 ``interview_node`` 的 ``is_interactive()``
  为 False，会自动用每题的最小风险默认值，不会卡在等待人工输入。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

# 只依赖 Helix 子图入口；刻意不 import askanswer.graph / persistence。
from .helix.agent import extract_helix_answer, run_helix_agent


# server 名尽量可读，便于外部 client 在工具前缀里识别来源。
mcp_server = FastMCP("askanswer-helix")


@mcp_server.tool()
def helix_spec_loop(topic: str) -> str:
    """规格优先开发循环：苏格拉底澄清 → 生成 Seed → 产出方案 → 自评演化。

    适用：给出模糊需求、希望先把“应该做什么”想清楚再动手（spec-first）。
    非交互环境（本 MCP server 即是）下澄清阶段自动采用最小风险默认假设。

    参数:
        topic: 用户的原始需求或想法描述（自然语言）。

    返回:
        Markdown 文本，含 Goal / Constraints / Acceptance criteria / Artifact /
        Evaluation / Lineage 六块。
    """
    result = run_helix_agent(topic)
    return extract_helix_answer(result)


def main() -> None:
    """以 stdio 传输启动 MCP server（默认传输，最通用）。"""
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
