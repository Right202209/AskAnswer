"""research 子图：检索规划收敛、经注册表检索并抽取来源、核验后带引用的简报。"""

from __future__ import annotations

import pytest

from askanswer.subagents.research import agent as research_agent
from askanswer.subagents.research import nodes as research_nodes

from .fakes import FakeStructuredModel


@pytest.mark.parametrize(
    "value,expected",
    [(None, 5), ("abc", 5), (0, 1), (-3, 1), (3, 3), (99, research_nodes.MAX_QUERIES)],
)
def test_clamp_queries(value, expected):
    assert research_nodes._clamp_queries(value) == expected


def test_extract_sources_dedupes_and_limits():
    text = "\n".join(f"   链接: https://e.com/{i % 7}" for i in range(20))
    sources = research_nodes._extract_sources(text)
    assert sources == [f"https://e.com/{i}" for i in range(5)]


class _FakeTavily:
    name = "tavily_search"

    def __init__(self):
        self.queries = []

    def invoke(self, args):
        query = args["query"]
        self.queries.append(query)
        return f"查询关键词：{query}\n1. **T** (相关度: 0.9)\n   链接: https://src/{query}\n   摘要"


class _Registry:
    def __init__(self, tool):
        self._tool = tool

    def get(self, name):
        if name != "tavily_search" or self._tool is None:
            return None

        class _Descriptor:
            tool = self._tool

        return _Descriptor()


def test_search_node_uses_registry_and_collects_sources(monkeypatch):
    tavily = _FakeTavily()
    monkeypatch.setattr("askanswer.tools.registry.get_registry", lambda: _Registry(tavily))
    out = research_nodes.search_node({"queries": ["a", "b"]})
    assert tavily.queries == ["a", "b"]
    assert [f["sources"] for f in out["findings"]] == [["https://src/a"], ["https://src/b"]]


def test_search_node_without_tavily_degrades(monkeypatch):
    monkeypatch.setattr("askanswer.tools.registry.get_registry", lambda: _Registry(None))
    out = research_nodes.search_node({"queries": ["a"]})
    assert "不可用" in out["findings"][0]["result"]
    assert out["findings"][0]["sources"] == []


def test_research_agent_end_to_end(monkeypatch):
    tavily = _FakeTavily()
    monkeypatch.setattr("askanswer.tools.registry.get_registry", lambda: _Registry(tavily))
    fake = FakeStructuredModel(
        {
            research_nodes.PlanOutput: research_nodes.PlanOutput(
                queries=["q1", " ", "q2", "q3"]
            ),
            research_nodes.SourceCheckOutput: lambda msgs: research_nodes.SourceCheckOutput(
                brief="结论 A [1]；结论 B [2]",
                references=["https://src/q1", " https://src/q2 ", ""],
            ),
        },
        texts=["草稿：结论 A [1]；结论 B [2]；推测 C"],
    )
    monkeypatch.setattr(research_nodes, "model", fake)
    result = research_agent.run_research_agent("最近的变化", max_queries=2)

    assert result["queries"] == ["q1", "q2"]  # 空白条目被丢弃，且收敛到 max_queries
    assert tavily.queries == ["q1", "q2"]
    assert result["references"] == ["https://src/q1", "https://src/q2"]
    # source_check 收到了编号后的来源清单
    source_prompt = fake.calls[-1][1][1].content
    assert "[1] https://src/q1" in source_prompt and "[2] https://src/q2" in source_prompt
    brief = research_agent.format_research_brief(result)
    assert "## References\n[1] https://src/q1\n[2] https://src/q2" in brief
    assert "## Queries\nq1 · q2" in brief
