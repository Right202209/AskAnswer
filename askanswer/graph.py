# 主图（父图）的拓扑：意图无关的三步骨架。
#   START → understand → answer → sorcery → (END | answer 重试)
# 真正的工具调用 / 多轮 react 在 answer 节点（react 子图）内部完成。
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import sorcery_answer_node, understand_query_node
from .react import build_react_subgraph
from .schema import ContextSchema
from .state import SearchState


def route_from_sorcery(state: SearchState):
    """sorcery 节点之后的条件分支：根据评估结果决定是否重新调用 answer 节点。"""
    # 当 sorcery 把 step 设为 retry_search 时，回到 answer 节点带着新关键词重试搜索
    if state["step"] == "retry_search":
        return "answer"
    # 其它情况一律结束（chat / sql / file_read 直接通过；search 通过或重试已用完）
    return END


def create_search_assistant(checkpointer: BaseCheckpointSaver | None = None):
    """构建并编译主图，返回可直接 invoke / stream 的 LangGraph 应用。

    :param checkpointer: 显式注入的 checkpointer。
        ``None`` 表示走默认持久化（``persistence.get_persistence().checkpointer``，
        即 ``~/.askanswer/state.db`` 上的 SqliteSaver）。
        想跳过持久化（如 ``--graph`` 导出 Mermaid、单测）时传 ``InMemorySaver()``。
    """
    # 父图共享 SearchState，并把 ContextSchema 作为运行时上下文 schema
    workflow = StateGraph(SearchState, context_schema=ContextSchema)
    # react 子图作为 answer 节点嵌入；它没有自己的 checkpointer，便于 interrupt 透传
    react = build_react_subgraph()

    # 三个节点：理解意图 → react 回答 → 评估/可能重搜
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("answer", react)
    workflow.add_node("sorcery", sorcery_answer_node)

    # 串联线性主流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "answer")
    workflow.add_edge("answer", "sorcery")
    # sorcery 之后通过 route_from_sorcery 决定继续重试还是结束
    workflow.add_conditional_edges(
        "sorcery",
        route_from_sorcery,
        {"answer": "answer", END: END},
    )

    if checkpointer is None:
        # 默认走持久化 SqliteSaver。延迟 import 避免 ``--graph`` 模式无意中
        # 触发 SQLite 文件创建（draw_search_assistant_mermaid 显式传 InMemorySaver）。
        from .persistence import get_persistence

        checkpointer = get_persistence().checkpointer

    return workflow.compile(checkpointer=checkpointer)


def draw_search_assistant_mermaid() -> str:
    """导出主图的 Mermaid 文本表示，供 ``askanswer --graph`` 使用。

    显式传 ``InMemorySaver`` 避免无谓地在 ``~/.askanswer/state.db`` 创建空 DB。
    """
    app = create_search_assistant(checkpointer=InMemorySaver())
    return app.get_graph().draw_mermaid()
