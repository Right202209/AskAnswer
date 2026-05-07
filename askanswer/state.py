# LangGraph 父图共享的状态定义。
# 所有节点的入参/返回值都基于 SearchState；新增持久化字段时记得在这里同步。
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class SearchState(TypedDict):
    # messages：对话历史。使用 add_messages 这个 reducer，节点返回的 messages 列表会被
    # 自动追加（而不是覆盖），保证多轮对话历史完整。
    messages: Annotated[list, add_messages]
    user_query: str      # 经过 LLM 理解后的用户需求总结（understand_query_node 写入）
    search_query: str    # 优化后用于 Tavily API 的搜索关键词
    search_results: str  # Tavily 搜索返回的结果文本
    final_answer: str    # 最终给用户的答案
    retry_count: int     # 已经触发重新搜索的次数（仅 search 路径会自增）
    step: str            # 当前流转到的步骤标记，例如 understood / tool_called / completed / retry_search
    intent: str          # 用户意图类别：file_read | search | chat | sql | math | 插件 intent
    file_path: str       # 当意图为 file_read 时，从用户输入提取出的目标文件路径
    retry_directive: dict # sorcery 给下一轮 answer 的结构化重试指令，answer 消费后清空
    pending_shell: dict  # {tool_call_id: {command, explanation, instruction}}，HITL（人机确认）时跨 interrupt 持久化
