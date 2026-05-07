# Helix 子图各节点：interview / seed / execute / evaluate / router。
# 每个节点都尽量薄；与 LLM 的交互全部走 model.with_structured_output 拿强类型，
# 避免再做正则解析。节点只返回部分状态字典，由 LangGraph 的 reducer 合并。
from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..load import model
from ..ui_select import CANCELLED, is_interactive, select_option
from .state import HelixState


# ── 单次工具调用允许的最大演化代数。SQL agent 的 MAX_SQL_QUERY_CALLS 同款思路：
#    防止评估反复打回导致 token 和时延双双爆掉。
MAX_GENERATIONS = 3


# ── Pydantic Schemas ────────────────────────────────────────────────

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


class SeedOutput(BaseModel):
    goal: str = Field(description="一句话清晰目标")
    constraints: list[str] = Field(default_factory=list, description="硬性约束")
    acceptance_criteria: list[str] = Field(
        description="可度量的验收条件，3-6 条", min_length=2, max_length=8
    )
    ontology: list[str] = Field(
        default_factory=list, description="核心实体/字段的简短列表"
    )
    principles: list[str] = Field(
        default_factory=list, description="评估原则，按重要性排序"
    )


class ExecuteOutput(BaseModel):
    artifact: str = Field(
        description="对应 Seed 的实现方案文本（步骤、关键代码骨架或配置示例）"
    )


class EvaluateOutput(BaseModel):
    verdict: Literal["approved", "rejected"] = Field(
        description="approved=可以收尾；rejected=需要继续演化"
    )
    score: float = Field(ge=0.0, le=1.0, description="语义对齐分数")
    gaps: list[str] = Field(
        default_factory=list,
        description="rejected 时必填，列出未覆盖的 acceptance_criteria 或漂移点",
    )


# ── helpers ─────────────────────────────────────────────────────────

def _structured(schema):
    """与 understand_query_node 同款的便捷封装。"""
    return model.with_structured_output(schema)


def _format_qa(qa: list[dict]) -> str:
    if not qa:
        return "（暂无 interview 记录）"
    return "\n".join(
        f"- [{item.get('track', '?')}] Q: {item.get('q', '')}\n  A: {item.get('a', '')}"
        for item in qa
    )


def _format_seed(seed: dict) -> str:
    if not seed:
        return "（暂无 seed）"
    lines = [f"goal: {seed.get('goal', '')}"]
    for key in ("constraints", "acceptance_criteria", "ontology", "principles"):
        values = seed.get(key) or []
        if values:
            lines.append(f"{key}:")
            lines.extend(f"  - {v}" for v in values)
    return "\n".join(lines)


# ── Nodes ───────────────────────────────────────────────────────────

# ANSI 颜色常量，仅用于 interview 阶段对用户的可视化提示。
_ORANGE = "\033[38;5;214m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[38;5;114m"
_RESET = "\033[0m"


def _ask_user(item: InterviewQA) -> str:
    """对单条 InterviewQA 询问用户：列出选项 + 一个“其他（手动输入）”入口。

    非交互环境（管道、CI）或用户取消时，回落到 LLM 提供的 ``default_answer``。
    """
    if not is_interactive():
        return item.default_answer

    print()
    print(
        f"  {_ORANGE}?{_RESET} {_BOLD}[{item.track}]{_RESET} "
        f"{item.question}"
    )
    idx, free_text = select_option(
        list(item.options),
        prompt=f"{_DIM}请选择最贴近你诉求的一项：{_RESET}",
        free_input_label="其他（手动输入）",
        free_input_prompt="自定义答案：",
    )
    if idx == CANCELLED:
        # 用户跳过：用 default_answer 作为最小风险假设。
        print(f"  {_DIM}已跳过，使用默认假设：{item.default_answer}{_RESET}")
        return item.default_answer
    if free_text is not None:
        # 命中“其他”：用户有自由输入则采用它，否则回落到默认。
        answer = free_text.strip() or item.default_answer
        print(f"  {_GREEN}✓ 你的回答：{_RESET}{answer}")
        return answer
    answer = item.options[idx]
    print(f"  {_GREEN}✓ 已选：{_RESET}{answer}")
    return answer


def interview_node(state: HelixState) -> dict:
    """苏格拉底式澄清：列出关键歧义并向用户征询答案（含选项 + 手动输入）。"""
    topic = state.get("topic") or ""
    system = SystemMessage(content=(
        "你是苏格拉底式访谈员。针对用户给出的模糊需求，列出最有杀伤力的关键问题。"
        "覆盖 scope/constraints/outputs/verification 四个歧义轨道，每轨道至少 1 条；"
        "为每个问题准备 2-5 个常见候选答案，让用户在 CLI 里挑选；"
        "另给一个最小风险的 default_answer（明确以 'assumption:' 开头标注），"
        "用于用户跳过或非交互环境时回落。"
    ))
    user = HumanMessage(content=f"用户需求：{topic}")
    output: InterviewOutput = _structured(InterviewOutput).invoke([system, user])

    if is_interactive():
        # 给用户一个清晰的 “开始访谈” 提示，避免选项菜单突兀地冒出来。
        print()
        print(f"  {_BOLD}{_ORANGE}Helix 苏格拉底访谈{_RESET}")
        print(
            f"  {_DIM}下面会问 {len(output.qa)} 个澄清问题；每题可上下选项，"
            f"也可挑“其他（手动输入）”自由作答。{_RESET}"
        )

    qa = []
    for item in output.qa:
        answer = _ask_user(item)
        qa.append({"track": item.track, "q": item.question, "a": answer})

    summary = f"Helix interview 产出 {len(qa)} 条 Q/A，覆盖 "
    summary += ", ".join(sorted({item['track'] for item in qa}))
    return {
        "interview_qa": qa,
        "messages": [AIMessage(content=summary)],
    }


def seed_node(state: HelixState) -> dict:
    """把 interview/上一代评估结果晶化为 Seed 规格。"""
    generation = state.get("generation") or 0
    next_gen = generation + 1
    seed_prev = state.get("seed") or {}
    evaluation = state.get("evaluation") or {}

    if next_gen == 1:
        directive = "首代：基于 interview Q/A 直接产出 Seed。"
    else:
        gaps = evaluation.get("gaps") or []
        directive = (
            f"第 {next_gen} 代：在保持核心目标的前提下修补以下 gaps：\n"
            + "\n".join(f"- {g}" for g in gaps)
            + "\n仅做最小必要变更，避免 ontology 漂移。"
        )

    system = SystemMessage(content=(
        "你是 Seed 架构师。把以下信息晶化成可被工程化的 Seed 规格。"
        "acceptance_criteria 必须是可度量的；constraints 限于硬性约束；"
        "ontology 用最简的实体列表表达。"
    ))
    user = HumanMessage(content=(
        f"用户需求：{state.get('topic', '')}\n\n"
        f"interview Q/A：\n{_format_qa(state.get('interview_qa') or [])}\n\n"
        f"演化指令：{directive}\n\n"
        f"上一代 Seed（如有）：\n{_format_seed(seed_prev)}"
    ))
    output: SeedOutput = _structured(SeedOutput).invoke([system, user])
    seed = output.model_dump()
    lineage = list(state.get("lineage") or [])
    lineage.append({"generation": next_gen, "seed": seed})
    return {
        "seed": seed,
        "lineage": lineage,
        "generation": next_gen,
        "messages": [AIMessage(content=f"Helix seed (gen {next_gen}) 已晶化：{seed['goal']}")],
    }


def execute_node(state: HelixState) -> dict:
    """根据 Seed 产出可被审阅的方案文本。"""
    seed = state.get("seed") or {}
    system = SystemMessage(content=(
        "你是工程实现者。仅根据 Seed 给出**贴合 acceptance_criteria** 的方案文本，"
        "可以包含步骤、关键代码骨架或配置示例。不要超出 Seed 约束范围。"
    ))
    user = HumanMessage(content=f"Seed：\n{_format_seed(seed)}")
    output: ExecuteOutput = _structured(ExecuteOutput).invoke([system, user])
    return {
        "artifact": output.artifact,
        "messages": [AIMessage(content="Helix execute 已产出方案。")],
    }


def evaluate_node(state: HelixState) -> dict:
    """三段式自评：覆盖率自查 + 语义对齐 + gaps。"""
    seed = state.get("seed") or {}
    artifact = state.get("artifact") or ""
    system = SystemMessage(content=(
        "你是评审。逐条核对 acceptance_criteria 是否被 artifact 覆盖；"
        "全部覆盖且无明显漂移给 verdict='approved'，否则 'rejected' 并列出 gaps。"
        "score 是 0–1 的语义对齐分数，rejected 时通常 < 0.7。"
    ))
    user = HumanMessage(content=(
        f"Seed：\n{_format_seed(seed)}\n\nArtifact：\n{artifact}"
    ))
    output: EvaluateOutput = _structured(EvaluateOutput).invoke([system, user])
    evaluation = output.model_dump()
    gen = state.get("generation") or 0
    return {
        "evaluation": evaluation,
        "messages": [AIMessage(
            content=(
                f"Helix evaluate (gen {gen}): {evaluation['verdict']} "
                f"score={evaluation['score']:.2f}"
            )
        )],
    }


def route_after_evaluate(
    state: HelixState,
) -> Literal["seed", "__end__"]:
    """approved → end；耗尽代数 → end；否则回到 seed 演化下一代。"""
    evaluation = state.get("evaluation") or {}
    if evaluation.get("verdict") == "approved":
        return "__end__"
    if (state.get("generation") or 0) >= MAX_GENERATIONS:
        return "__end__"
    return "seed"


def finalize_verdict(state: HelixState) -> dict:
    """END 前的小整理节点：把最终 verdict 写实，方便上层读取。"""
    evaluation = state.get("evaluation") or {}
    if evaluation.get("verdict") == "approved":
        verdict = "approved"
    elif (state.get("generation") or 0) >= MAX_GENERATIONS:
        verdict = "exhausted"
    else:
        verdict = "rejected"
    return {"verdict": verdict}
