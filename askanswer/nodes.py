from langchain.messages import ToolMessage
from langchain_core.messages import AIMessage, SystemMessage

from .load import model, tavily_client
from .state import SearchState
from .tools import tools_by_name


def understand_query_node(state: SearchState) -> dict:
    user_message = state["messages"][-1].content
    understand_prompt = f"""分析用户的查询："{user_message}"
    请完成两个任务：
    1. 简洁总结用户想要了解什么
    2. 生成最适合搜索引擎的关键词（中英文均可，要精准）
    格式：
    理解：[用户需求总结]
    搜索词：[最佳搜索关键词]"""

    response = model.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content

    search_query = user_message  # 默认使用原始查询
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()

    return {
        "user_query": response_text,
        "search_query": search_query,
        "retry_count": 0,
        "step": "understood",
        "messages": [AIMessage(content=f"我将为您搜索：{search_query}")],
    }


def tavily_search_node(state: SearchState) -> dict:
    search_query = state["search_query"]
    try:
        print(f"正在搜索：{search_query}")
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )

        results = response.get("results", [])

        search_results = f"查询关键词：{search_query}\n\n"
        search_results += "搜索结果（Top 5）：\n\n"

        if results:
            for i, result in enumerate(results, 1):
                title = result.get("title", "无标题")
                url = result.get("url", "#")
                content = result.get("content", "无内容摘要")
                score = result.get("score", 0.0)

                search_results += (
                    f"{i}. **{title}** (相关度: {score:.3f})\n"
                    f"   链接: {url}\n"
                    f"   {content[:280]}{'...' if len(content) > 280 else ''}\n\n"
                )
        else:
            search_results += "未找到任何搜索结果。\n"

        return {
            "search_results": search_results.strip(),
            "step": "searched",
            "messages": [AIMessage(content="搜索完成~ 正在整理答案...")],
        }
    except Exception as e:
        error_msg = f"搜索失败：{str(e)}"
        print(f"Tavily 搜索异常: {error_msg}")

        return {
            "search_results": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": [AIMessage(content=" 搜索遇到问题...")],
        }


def generate_answer_node(state: SearchState) -> dict:
    if state["step"] == "search_failed":
        fallback_prompt = (
            f"搜索API暂时不可用，请基于您的知识回答用户的问题：\n用户问题：{state['user_query']}"
        )
        response = model.invoke([SystemMessage(content=fallback_prompt)])
    else:
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：
        用户问题：{state['user_query']}
        搜索结果：\n{state['search_results']}
        请综合搜索结果，提供准确、有用的回答..."""
        response = model.invoke([SystemMessage(content=answer_prompt)])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)],
    }


def sorcery_answer_node(state: SearchState) -> dict:
    retry_count = state.get("retry_count", 0)
    if retry_count >= 1:
        final_answer = state.get("final_answer", "")
        return {
            "final_answer": final_answer,
            "step": "completed",
            "messages": [AIMessage(content=final_answer)],
        }

    prompt = f"""请评估当前答案是否足够好。
    用户问题：{state.get('user_query', '')}
    当前搜索词：{state.get('search_query', '')}
    当前搜索结果：{state.get('search_results', '')}
    当前答案：{state.get('final_answer', '')}

    如果答案已经足够好，请严格输出：
    评分：pass
    新搜索词：

    如果答案不够好，需要重新搜索，请严格输出：
    评分：re_search
    新搜索词：[更精准的新搜索词]
    """
    response = model.invoke([SystemMessage(content=prompt)])
    response_text = response.content

    if "评分：re_search" in response_text and "新搜索词：" in response_text:
        new_search_query = response_text.split("新搜索词：", 1)[1].strip()
        if new_search_query:
            return {
                "search_query": new_search_query,
                "retry_count": retry_count + 1,
                "step": "retry_search",
                "messages": [
                    AIMessage(content=f"当前答案不够理想，改为搜索：{new_search_query}")
                ],
            }

    final_answer = state.get("final_answer", "")
    return {
        "final_answer": final_answer,
        "step": "completed",
        "messages": [AIMessage(content=final_answer)],
    }


def tools_node(state: SearchState) -> dict:
    res = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        res.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": res}
