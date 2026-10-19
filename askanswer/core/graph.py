"""Parent graph topology: an intent-agnostic three-step skeleton.

``START → understand → answer → sorcery → (END | answer retry)``

Tool use and the multi-turn ReAct loop happen inside the ``answer`` node (the
ReAct subgraph); intents never fork the parent graph.
(父图拓扑：意图无关的三步骨架；工具调用在 answer 节点的 react 子图内部完成。)
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ..config.schema import ContextSchema
from ..config.settings import ensure_bootstrapped
from .nodes import sorcery_answer_node, understand_query_node
from .react import build_react_subgraph
from .state import SearchState


def route_from_sorcery(state: SearchState):
    """sorcery 节点之后的条件分支：根据评估结果决定是否重新调用 answer 节点。"""
    # 当 sorcery 把 step 设为 retry_search 时，回到 answer 节点带着新关键词重试搜索
    if state["step"] == "retry_search":
        return "answer"
    # 其它情况一律结束（chat / sql / file_read 直接通过；search 通过或重试已用完）
    return END


def create_search_assistant(checkpointer: BaseCheckpointSaver | None = None):
    """Build and compile the agent graph; returns a LangGraph app (invoke / stream).

    :param checkpointer: checkpointer to compile with. ``None`` uses the default
        SQLite persistence (``askanswer.storage.get_persistence().checkpointer``, i.e.
        ``SqliteSaver`` on ``~/.askanswer/state.db``). Pass any other saver, e.g.
        ``InMemorySaver()``, to stay off disk; :func:`askanswer.run_leg` then writes
        audit / thread metadata to ``NullPersistence`` as well.

    The first call lazily bootstraps configuration (``.env`` + ``settings.json``).
    """
    # 首次构图时惰性加载 settings.json / .env（导入期不碰 os.environ）
    ensure_bootstrapped()
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
        # 默认走持久化 SqliteSaver；延迟 import，避免模块导入时创建 SQLite 文件。
        from ..storage import get_persistence

        checkpointer = get_persistence().checkpointer

    return workflow.compile(checkpointer=checkpointer)
