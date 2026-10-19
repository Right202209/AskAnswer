"""Shared Socratic interview node for clarification-first subagents.

Both :mod:`askanswer.subagents.helix` and :mod:`askanswer.subagents.decision`
start with this node: the model lists the key questions across four ambiguity
tracks (scope / constraints / outputs / verification) and, because subagents run
headless (no UI inside a tool call), each question's lowest-risk
``default_answer`` (prefixed ``assumption:``) is adopted directly.

The node reads ``state["topic"]`` and writes ``interview_qa`` (list of
``{"track", "q", "a"}``) plus one summary ``AIMessage``; any state with those keys
(see :class:`InterviewState`) can embed it.

Fallback: interviewing is a best-effort refinement.  If the model call or the
structured parse fails, the node records an empty interview with an explicit
``assumption:`` note and lets the subgraph continue from the raw topic instead of
failing the whole tool call.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from ...llm.proxy import model


class InterviewState(MessagesState):
    """Minimal state contract of :func:`interview_node`."""

    topic: str
    interview_qa: list[dict]


class InterviewQA(BaseModel):
    track: Literal["scope", "constraints", "outputs", "verification"] = Field(
        description="问题所属的歧义轨道"
    )
    question: str = Field(description="苏格拉底式问题，单一焦点")
    options: list[str] = Field(
        description="给用户挑选的候选答案，覆盖最常见取舍；2-5 条最佳",
        min_length=2,
        max_length=5,
    )
    default_answer: str = Field(
        description=(
            "若用户跳过该问题（非交互环境或主动取消），用此作为最小风险默认值；"
            "需明确以 'assumption:' 开头标注其为假设。"
        )
    )


class InterviewOutput(BaseModel):
    qa: list[InterviewQA] = Field(
        description="覆盖四个歧义轨道，每轨道至少一条；3-6 条最佳",
        min_length=3,
        max_length=8,
    )


def format_qa(qa: list[dict]) -> str:
    """Render interview Q/A pairs as a bullet list for downstream prompts."""
    if not qa:
        return "（暂无 interview 记录）"
    return "\n".join(
        f"- [{item.get('track', '?')}] Q: {item.get('q', '')}\n  A: {item.get('a', '')}"
        for item in qa
    )


def interview_node(state: InterviewState) -> dict:
    """生成关键澄清问题，并在无 UI 的 agent 运行中采用最小风险假设。"""
    topic = state.get("topic") or ""
    system = SystemMessage(content=(
        "你是苏格拉底式访谈员。针对用户给出的模糊需求，列出最有杀伤力的关键问题。"
        "覆盖 scope/constraints/outputs/verification 四个歧义轨道，每轨道至少 1 条；"
        "为每个问题准备 2-5 个常见候选答案；"
        "另给一个最小风险的 default_answer（明确以 'assumption:' 开头标注），"
        "用于当前无 UI 的运行路径直接采用。"
    ))
    user = HumanMessage(content=f"用户需求：{topic}")
    try:
        output: InterviewOutput = model.with_structured_output(InterviewOutput).invoke(
            [system, user]
        )
    except Exception as exc:
        # 访谈只是锦上添花：失败时以原始需求为准继续，而不是让整个子图崩掉
        return {
            "interview_qa": [],
            "messages": [AIMessage(content=(
                f"interview 跳过（{type(exc).__name__}）；"
                "assumption: 直接按用户原始需求推进，不做额外澄清。"
            ))],
        }

    qa = []
    for item in output.qa:
        qa.append({"track": item.track, "q": item.question, "a": item.default_answer})

    summary = f"interview 产出 {len(qa)} 条 Q/A，覆盖 "
    summary += ", ".join(sorted({item['track'] for item in qa}))
    return {
        "interview_qa": qa,
        "messages": [AIMessage(content=summary)],
    }
