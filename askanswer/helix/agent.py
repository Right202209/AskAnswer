# Helix 子图：把 interview → seed → execute → evaluate 演化循环编译成独立子图。
# 与 SQL agent 同样的封装思路：build_*  返回编译好的 Graph，run_*  是统一入口，
# 通过硬上限 MAX_GENERATIONS / RECURSION_LIMIT 防止循环失控。
from __future__ import annotations

from langchain_core.messages import HumanMessage

from langgraph.graph import END, START, StateGraph

from ..schema import ContextSchema, normalize_context
from .nodes import (
    MAX_GENERATIONS,
    evaluate_node,
    execute_node,
    finalize_verdict,
    interview_node,
    route_after_evaluate,
    seed_node,
)
from .state import HelixState


# 递归上限：3 代 × 4 节点 + 收尾 + 余量，给 LangGraph 兜底。
RECURSION_LIMIT = 24


def build_helix_agent():
    """构建并编译 Helix 子图：interview → seed → execute → evaluate (→ seed)*。"""
    builder = StateGraph(HelixState, context_schema=ContextSchema)

    builder.add_node("interview", interview_node)     # 苏格拉底澄清
    builder.add_node("seed", seed_node)               # 晶化 Seed
    builder.add_node("execute", execute_node)         # 产出 artifact
    builder.add_node("evaluate", evaluate_node)       # 自评
    builder.add_node("finalize", finalize_verdict)    # 收尾写 verdict

    builder.add_edge(START, "interview")
    builder.add_edge("interview", "seed")
    builder.add_edge("seed", "execute")
    builder.add_edge("execute", "evaluate")
    # evaluate 之后：approved/exhausted → finalize；否则回到 seed 演化下一代
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"seed": "seed", END: "finalize"},
    )
    builder.add_edge("finalize", END)

    return builder.compile()


# 模块级单例：编译一次，多次复用。
helix_agent = build_helix_agent()


def run_helix_agent(topic: str, context: ContextSchema | dict | None = None) -> dict:
    """执行 Helix 子图并返回最终状态字典。

    入参 ``topic`` 即用户原始的模糊需求；``context`` 透传 ContextSchema 给子图，
    便于未来在节点里读取 db_dsn / tenant_id 等 per-invocation 配置。
    """
    initial: HelixState = {
        "messages": [HumanMessage(content=topic)],
        "topic": topic,
        "interview_qa": [],
        "seed": {},
        "artifact": "",
        "evaluation": {},
        "generation": 0,
        "lineage": [],
        "verdict": "",
    }
    result = helix_agent.invoke(
        initial,
        config={"recursion_limit": RECURSION_LIMIT},
        context=normalize_context(context),
    )
    return result


def format_helix_summary(result: dict) -> str:
    """把子图结果整理成 Markdown 摘要，作为 helix_spec_loop 工具的最终返回。"""
    seed = result.get("seed") or {}
    evaluation = result.get("evaluation") or {}
    lineage = result.get("lineage") or []
    verdict = result.get("verdict") or "unknown"
    artifact = result.get("artifact") or "（未产出）"

    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {x}" for x in (items or [])) or "-"

    score = evaluation.get("score")
    score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
    gaps_text = _bullets(evaluation.get("gaps") or [])
    lineage_text = " → ".join(
        f"gen{item.get('generation', '?')}" for item in lineage
    ) or "（无）"

    return (
        f"## Goal\n{seed.get('goal', '（未生成）')}\n\n"
        f"## Constraints\n{_bullets(seed.get('constraints'))}\n\n"
        f"## Acceptance criteria\n{_bullets(seed.get('acceptance_criteria'))}\n\n"
        f"## Artifact\n{artifact}\n\n"
        f"## Evaluation\n"
        f"- verdict: **{verdict}**\n"
        f"- score: {score_str}\n"
        f"- gaps:\n{gaps_text}\n\n"
        f"## Lineage\n{lineage_text}（最大 {MAX_GENERATIONS} 代）"
    )


def extract_helix_answer(result: dict) -> str:
    """对外暴露的最终答案字符串：直接复用 Markdown 摘要。"""
    return format_helix_summary(result)


__all__ = [
    "MAX_GENERATIONS",
    "RECURSION_LIMIT",
    "build_helix_agent",
    "helix_agent",
    "run_helix_agent",
    "format_helix_summary",
    "extract_helix_answer",
]
