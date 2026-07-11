"""把 decision 子图封装成 LLM 可调用的 Tool。

与 helix_tool / research_tool 同款：被 react 子图的 ``ToolNode`` 调用时 ``runtime`` 会
自动注入，子图内部就能拿到与父图一致的 ``ContextSchema``。
"""

from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ..schema import ContextSchema, normalize_context
from .agent import extract_decision_answer, run_decision_agent


@tool
def decision_memo_loop(topic: str, runtime: ToolRuntime[ContextSchema]) -> str:
    """决策备忘循环：苏格拉底澄清目标/约束 → 输出候选方案的取舍分析与推荐。

    适用：用户面临“在几个方向里选一个”的决策，希望先澄清目标与约束，再拿到带
    pros/cons 的对比与明确推荐。非交互环境下自动用每个澄清问题的最小风险默认值。

    参数:
        topic: 用户要做的决策（自然语言）。

    返回:
        Markdown 文本，含 Goal / Constraints / Options / Recommendation 四块。
    """
    context = normalize_context(getattr(runtime, "context", None))
    result = run_decision_agent(topic, context=context)
    return extract_decision_answer(result)
