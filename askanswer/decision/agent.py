"""decision_memo 子图：澄清目标/约束 → 输出 tradeoff memo。

复用 Helix 的 ``interview_node`` 做前置苏格拉底澄清（非 TTY 环境自动回退到每题的
``default_answer``），再由 ``decide_node`` 晶化成带取舍分析的决策备忘。刻意**不改父图**，
仅作为 subgraph-as-tool 暴露；也不在父 sorcery 里重试（``max_retries=0``）。
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from ..helix.nodes import interview_node
from ..load import model
from ..schema import ContextSchema, normalize_context


# 决策备忘里方案数量的软上限，避免 LLM 铺开太多选项稀释重点。
MAX_OPTIONS = 4


class DecisionState(MessagesState):
    """decision 子图状态：主题 + interview 澄清 Q/A + 最终 memo。"""
    topic: str
    interview_qa: list[dict]
    memo: dict


class DecisionOption(BaseModel):
    name: str = Field(description="方案名称")
    pros: list[str] = Field(default_factory=list, description="该方案的主要优点")
    cons: list[str] = Field(default_factory=list, description="该方案的主要代价/风险")


class MemoOutput(BaseModel):
    goal: str = Field(description="一句话决策目标")
    constraints: list[str] = Field(default_factory=list, description="硬性约束")
    options: list[DecisionOption] = Field(
        description="候选方案的取舍分析，2-4 个", min_length=1, max_length=MAX_OPTIONS
    )
    recommendation: str = Field(description="推荐方案名（须出现在 options 里）")
    rationale: str = Field(description="推荐理由：为什么在给定目标/约束下它最优")


def _format_qa(qa: list[dict]) -> str:
    if not qa:
        return "（无 interview 记录）"
    return "\n".join(
        f"- [{item.get('track', '?')}] Q: {item.get('q', '')}\n  A: {item.get('a', '')}"
        for item in qa
    )


def decide_node(state: DecisionState) -> dict:
    """基于主题与 interview 澄清，产出带取舍分析的决策备忘。"""
    system = SystemMessage(content=(
        "你是决策顾问。基于用户目标与澄清答案，给出 2-4 个候选方案的取舍分析"
        "（各列优点与代价），再给出明确推荐与理由。推荐必须落在候选方案之内，"
        "不要模棱两可。"
    ))
    user = HumanMessage(content=(
        f"决策主题：{state.get('topic', '')}\n\n"
        f"澄清 Q/A：\n{_format_qa(state.get('interview_qa') or [])}"
    ))
    output: MemoOutput = model.with_structured_output(MemoOutput).invoke([system, user])
    memo = output.model_dump()
    return {
        "memo": memo,
        "messages": [AIMessage(content=f"decision memo 已产出，推荐：{memo['recommendation']}")],
    }


def build_decision_agent():
    """构建并编译 decision 子图：interview → decide。"""
    builder = StateGraph(DecisionState, context_schema=ContextSchema)
    builder.add_node("interview", interview_node)   # 复用 Helix 苏格拉底澄清
    builder.add_node("decide", decide_node)         # 晶化 tradeoff memo
    builder.add_edge(START, "interview")
    builder.add_edge("interview", "decide")
    builder.add_edge("decide", END)
    return builder.compile()


# 模块级单例：编译一次，多次复用。
decision_agent = build_decision_agent()


def run_decision_agent(topic: str, context: ContextSchema | dict | None = None) -> dict:
    """执行 decision 子图并返回最终状态字典。"""
    initial: DecisionState = {
        "messages": [HumanMessage(content=topic)],
        "topic": topic,
        "interview_qa": [],
        "memo": {},
    }
    return decision_agent.invoke(initial, context=normalize_context(context))


def format_decision_memo(result: dict) -> str:
    """把子图结果整理成 Markdown 决策备忘。"""
    memo = result.get("memo") or {}
    goal = memo.get("goal") or "（未生成）"
    constraints = memo.get("constraints") or []
    options = memo.get("options") or []
    recommendation = memo.get("recommendation") or "（无）"
    rationale = memo.get("rationale") or "（无）"

    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {x}" for x in (items or [])) or "-"

    option_blocks = []
    for opt in options:
        option_blocks.append(
            f"### {opt.get('name', '(未命名)')}\n"
            f"**Pros**\n{_bullets(opt.get('pros'))}\n\n"
            f"**Cons**\n{_bullets(opt.get('cons'))}"
        )
    options_text = "\n\n".join(option_blocks) or "（无候选方案）"

    return (
        f"## Goal\n{goal}\n\n"
        f"## Constraints\n{_bullets(constraints)}\n\n"
        f"## Options\n{options_text}\n\n"
        f"## Recommendation\n**{recommendation}**\n\n{rationale}"
    )


def extract_decision_answer(result: dict) -> str:
    """对外暴露的最终答案字符串：直接复用 Markdown 备忘。"""
    return format_decision_memo(result)


__all__ = [
    "MAX_OPTIONS",
    "build_decision_agent",
    "decision_agent",
    "run_decision_agent",
    "format_decision_memo",
    "extract_decision_answer",
]
