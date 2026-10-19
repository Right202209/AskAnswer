"""Helix / decision：共享访谈节点（默认答案采纳 + 失败兜底）与演化循环终止。"""

from __future__ import annotations

import pytest

from askanswer.subagents.common import interview
from askanswer.subagents.common.interview import InterviewOutput, InterviewQA
from askanswer.subagents.decision import agent as decision_agent
from askanswer.subagents.helix import agent as helix_agent
from askanswer.subagents.helix import nodes as helix_nodes

from .fakes import FakeStructuredModel


def _interview_output() -> InterviewOutput:
    return InterviewOutput(qa=[
        InterviewQA(track="scope", question="做给谁？", options=["自己", "团队"],
                    default_answer="assumption: 只给自己用"),
        InterviewQA(track="constraints", question="预算？", options=["0", "少量"],
                    default_answer="assumption: 零预算"),
        InterviewQA(track="verification", question="怎么验收？", options=["手测", "自动"],
                    default_answer="assumption: 手工验收"),
    ])


def _seed():
    return helix_nodes.SeedOutput(goal="做一个待办应用", constraints=["零预算"],
                                  acceptance_criteria=["能添加待办", "能删除待办"])


# ── 共享访谈节点 ─────────────────────────────────────────────────────────

def test_interview_adopts_default_answers(monkeypatch):
    fake = FakeStructuredModel({InterviewOutput: _interview_output()})
    monkeypatch.setattr(interview, "model", fake)
    out = interview.interview_node({"topic": "想做个应用", "messages": []})
    assert [qa["a"] for qa in out["interview_qa"]] == [
        "assumption: 只给自己用", "assumption: 零预算", "assumption: 手工验收",
    ]
    assert {qa["track"] for qa in out["interview_qa"]} == {"scope", "constraints", "verification"}
    assert "3 条" in out["messages"][0].content


def test_interview_falls_back_when_model_fails(monkeypatch):
    fake = FakeStructuredModel({InterviewOutput: RuntimeError("provider down")})
    monkeypatch.setattr(interview, "model", fake)
    out = interview.interview_node({"topic": "想做个应用", "messages": []})
    assert out["interview_qa"] == []
    assert "assumption:" in out["messages"][0].content


def test_format_qa_shared_by_helix_and_decision():
    text = interview.format_qa([{"track": "scope", "q": "Q1", "a": "A1"}])
    assert text == "- [scope] Q: Q1\n  A: A1"
    assert interview.format_qa([]) == "（暂无 interview 记录）"


def test_decision_and_helix_share_interview_node():
    assert decision_agent.interview_node is interview.interview_node
    assert helix_agent.interview_node is interview.interview_node


# ── Helix 演化循环终止 ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    "evaluation,generation,expected",
    [
        ({"verdict": "approved"}, 1, "__end__"),
        ({"verdict": "rejected"}, 1, "seed"),
        ({"verdict": "rejected"}, helix_nodes.MAX_GENERATIONS, "__end__"),
    ],
)
def test_route_after_evaluate(evaluation, generation, expected):
    state = {"evaluation": evaluation, "generation": generation}
    assert helix_nodes.route_after_evaluate(state) == expected


def test_helix_loop_stops_after_max_generations(monkeypatch):
    rejected = helix_nodes.EvaluateOutput(verdict="rejected", score=0.3, gaps=["缺少删除"])
    fake = FakeStructuredModel({
        InterviewOutput: _interview_output(),
        helix_nodes.SeedOutput: _seed(),
        helix_nodes.ExecuteOutput: helix_nodes.ExecuteOutput(artifact="方案 v1"),
        helix_nodes.EvaluateOutput: rejected,
    })
    monkeypatch.setattr(interview, "model", fake)
    monkeypatch.setattr(helix_nodes, "model", fake)
    result = helix_agent.run_helix_agent("想做个待办应用")
    assert result["generation"] == helix_nodes.MAX_GENERATIONS
    assert result["verdict"] == "exhausted"
    assert len(result["lineage"]) == helix_nodes.MAX_GENERATIONS
    summary = helix_agent.format_helix_summary(result)
    assert "verdict: **exhausted**" in summary and "gen1 → gen2 → gen3" in summary


def test_helix_loop_approves_first_generation(monkeypatch):
    fake = FakeStructuredModel({
        InterviewOutput: RuntimeError("interview down"),  # 访谈失败也能走完
        helix_nodes.SeedOutput: _seed(),
        helix_nodes.ExecuteOutput: helix_nodes.ExecuteOutput(artifact="方案"),
        helix_nodes.EvaluateOutput: helix_nodes.EvaluateOutput(verdict="approved", score=0.9),
    })
    monkeypatch.setattr(interview, "model", fake)
    monkeypatch.setattr(helix_nodes, "model", fake)
    result = helix_agent.run_helix_agent("想做个待办应用")
    assert result["verdict"] == "approved"
    assert result["generation"] == 1
    assert result["interview_qa"] == []


# ── decision memo ────────────────────────────────────────────────────────

def test_decision_agent_produces_memo(monkeypatch):
    memo = decision_agent.MemoOutput(
        goal="选一个数据库",
        constraints=["团队熟悉 SQL"],
        options=[
            decision_agent.DecisionOption(name="PostgreSQL", pros=["成熟"], cons=["运维"]),
            decision_agent.DecisionOption(name="SQLite", pros=["零运维"], cons=["并发弱"]),
        ],
        recommendation="PostgreSQL",
        rationale="并发需求高",
    )
    fake = FakeStructuredModel({InterviewOutput: _interview_output(),
                                decision_agent.MemoOutput: memo})
    monkeypatch.setattr(interview, "model", fake)
    monkeypatch.setattr(decision_agent, "model", fake)
    result = decision_agent.run_decision_agent("PostgreSQL 还是 SQLite？")
    text = decision_agent.format_decision_memo(result)
    assert "## Recommendation\n**PostgreSQL**" in text
    assert "### SQLite" in text
    # decide 节点拿到了访谈的默认答案
    decide_prompt = fake.calls[-1][1][1].content
    assert "assumption: 零预算" in decide_prompt
