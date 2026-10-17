"""P2：依赖经 Container 注入——整张主图用替身跑通，不 monkeypatch 任何模块全局。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from askanswer.config import load_settings
from askanswer.graph import create_search_assistant
from askanswer.graph.context import ContextSchema
from askanswer.intents.base import IntentClassification
from askanswer.runtime import build_container
from askanswer.tools.registry import ToolRegistry


class _FakeModel:
    label = "fake:model"

    def __init__(self):
        self.calls: list[str] = []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        outer = self

        class _S:
            def invoke(self, messages):
                outer.calls.append("classify")
                return IntentClassification(intent="chat", understanding="打招呼")

        return _S()

    def invoke(self, messages):
        self.calls.append("answer")
        return AIMessage(content="你好！")


def test_parent_graph_runs_on_injected_dependencies():
    model = _FakeModel()
    container = build_container(
        settings=load_settings({}),
        tools=ToolRegistry(),          # 空注册表：不 seed 内置工具、不连 MCP
        models=lambda role: model,
    )
    app = create_search_assistant(InMemorySaver(), container=container)
    cfg = {"configurable": {"thread_id": "c1"}}
    out = app.invoke(
        {"messages": [HumanMessage(content="嗯嗯嗯")]}, config=cfg, context=ContextSchema()
    )
    assert out["final_answer"] == "你好！"
    assert out["intent"] == "chat"
    assert out["step"] == "completed"
    assert "answer" in model.calls
    # P6：历史里只有用户消息和一条模型回答
    assert [m.type for m in out["messages"]] == ["human", "ai"]


def test_container_resolves_defaults_lazily():
    seen = []

    def tools_provider():
        seen.append("tools")
        return ToolRegistry()

    from askanswer.runtime.container import Container

    container = Container(
        tools_provider=tools_provider,
        intents_provider=lambda: None,
        models=lambda role: None,
        confirmations=lambda clazz: None,
    )
    create_search_assistant(InMemorySaver(), container=container)
    assert seen == []  # 建图不触发工具注册表构造
    assert isinstance(container.tools, ToolRegistry)


def test_injected_settings_are_used():
    settings = load_settings({"ASKANSWER_RUN_TOKEN_BUDGET": "5"})
    container = build_container(settings=settings)
    assert container.settings.run_token_budget == 5
