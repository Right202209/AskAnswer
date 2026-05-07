"""把 Helix 子图封装成 LLM 可调用的 Tool。

与 sql_tool 同款思路：在 react 子图里被 ``ToolNode`` 调用时，``runtime`` 参数
会自动注入，子图内部就能拿到与父图一致的 ``ContextSchema``。
"""

from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ..schema import ContextSchema, normalize_context
from .agent import extract_helix_answer, run_helix_agent


@tool
def helix_spec_loop(topic: str, runtime: ToolRuntime[ContextSchema]) -> str:
    """规格优先开发循环：苏格拉底澄清 → 生成 Seed → 产出方案 → 自评演化。

    适用：用户给出模糊需求、希望先把"应该做什么"想清楚再"开始做"，或者
    需要从需求层面迭代演化（spec-first / specification-first / evolutionary loop）。

    参数:
        topic: 用户的原始需求或想法描述（自然语言）。

    返回:
        Markdown 文本，包含 Goal / Constraints / Acceptance criteria /
        Artifact / Evaluation / Lineage 六块。
    """
    context = normalize_context(getattr(runtime, "context", None))
    result = run_helix_agent(topic, context=context)
    return extract_helix_answer(result)
