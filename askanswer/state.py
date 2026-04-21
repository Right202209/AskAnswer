from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str      # 经过LLM理解后的用户需求总结
    search_query: str    # 优化后用于Tavily API的搜索查询
    search_results: str  # Tavily搜索返回的结果
    final_answer: str    # 最终生成的答案
    retry_count: int     # 重新搜索次数
    step: str            # 标记当前步骤
    intent: str          # file_read | search | chat
    file_path: str       # 当意图为 file_read 时的目标路径
    pending_shell: dict  # {tool_call_id: {command, explanation, instruction}} 供 HITL 确认使用
