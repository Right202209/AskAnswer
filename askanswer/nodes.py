# 父图前置节点：意图分类（understand）和事后评估（sorcery）。
# 设计原则：先用本地正则/关键词做高置信度分类，避免每次都调用 LLM；
# 仅当本地启发不确定时才回退到 LLM 来判定 intent。

from langchain_core.messages import AIMessage, SystemMessage

from .intents import get_intent_registry
from .intents.base import IntentClassification
from .load import model
from .state import SearchState


def _local_intent(user_message: str, *, fallback: bool = False) -> dict | None:
    """高置信度的本地意图分类器。

    返回 None 表示“说不清楚”，由调用方决定是否走 LLM；当 ``fallback=True`` 时
    则强制给出一个尽量合理的 intent，作为 LLM 失败时的兜底。
    """
    fields = get_intent_registry().classify_local(user_message, fallback=fallback)
    if fields is None:
        return None
    return fields.model_dump()


def _normalize_intent(fields: dict, user_message: str) -> dict:
    """把任何来源的 intent dict 归一化成标准格式（统一字段、补默认值）。"""
    return get_intent_registry().normalize(fields, user_message).model_dump()


def _intent_from_llm(user_message: str) -> dict:
    """调用 LLM 做意图分类，使用结构化输出约束字段。"""
    registry = get_intent_registry()
    intent_list = registry.llm_intent_list()
    prompt = (
        f'分析用户的查询："{user_message}"\n\n'
        "请判断用户意图属于以下哪一类：\n"
        "- file_read：要求读取或分析本地文件（通常会给出路径或文件名，"
        "如 /tmp/a.txt、./data.csv、report.md）\n"
        "- sql：要求查询数据库、编写 SQL、分析表结构、统计数据库数据\n"
        "- math：明确要求数学表达式计算\n"
        "- chat：闲聊、常识性问题，或你已知可直接回答、不需要联网\n"
        "- search：需要联网搜索获取实时、最新或不确定的信息\n\n"
        f"intent 必须是这些值之一：{intent_list}。\n"
        "file_path 仅 file_read 时填写具体路径，否则为空字符串。\n"
        "search_query 仅 search 时填写最佳关键词，否则为空字符串。\n"
        "understanding 填写对用户需求的简要总结。"
    )
    classifier = model.with_structured_output(IntentClassification)
    response = classifier.invoke([SystemMessage(content=prompt)])
    return _normalize_intent(response.model_dump(), user_message)


def understand_query_node(state: SearchState) -> dict:
    """父图第一个节点：把用户最新消息分类为 file_read/sql/chat/search。"""
    user_message = state["messages"][-1].content
    # 优先用本地分类器
    fields = _local_intent(user_message)
    if fields is None:
        # 不确定时再调用 LLM；LLM 也失败就走 fallback 强制给个意图，避免流程卡死
        try:
            fields = _intent_from_llm(user_message)
        except Exception:
            fields = _local_intent(user_message, fallback=True)

    intent = fields["intent"]
    search_query = fields["search_query"]
    file_path = fields["file_path"]
    understanding = fields["understanding"]

    if intent == "file_read":
        hint = f"识别为读文件：{file_path or '(未能提取路径)'}"
    elif intent == "sql":
        hint = "识别为数据库/SQL 问题，转交 sql_query 工具"
    elif intent == "math":
        hint = "识别为数学计算问题，转交 calculate 工具"
    elif intent == "chat":
        hint = "识别为闲聊/常识问题，直接回答"
    else:
        hint = f"识别为搜索，关键词：{search_query}"

    # 注意：节点返回的是“部分状态”字典，会被 LangGraph 合并到 SearchState
    return {
        "user_query": understanding,
        "search_query": search_query,
        "file_path": file_path,
        "intent": intent,
        "retry_count": 0,
        "retry_directive": {},
        "step": "understood",
        "messages": [AIMessage(content=hint)],
    }


def sorcery_answer_node(state: SearchState) -> dict:
    """父图第三个节点：把答案质量评估委托给当前 intent handler。"""
    handler = get_intent_registry().get(state.get("intent", "search"))

    retry_count = state.get("retry_count", 0)
    if retry_count >= handler.max_retries:
        return _finalize(state)

    result = handler.evaluate(state)
    if result.decision == "pass":
        return _finalize(state)

    retry_directive = result.retry_directive or {}
    search_query = retry_directive.get("search_query")
    message = result.reason or "当前答案不够理想，准备重试"
    out = {
        "retry_count": retry_count + 1,
        "retry_directive": retry_directive,
        "step": "retry_search",
        "messages": [AIMessage(content=message)],
    }
    if search_query:
        out["search_query"] = search_query
    return out


def _finalize(state: SearchState) -> dict:
    """sorcery 评估通过 / 重试预算耗尽后的统一收尾：清空 retry_directive 并打 step。"""
    final_answer = state.get("final_answer", "")
    return {
        "final_answer": final_answer,
        "step": "completed",
        "retry_directive": {},
        "messages": [AIMessage(content=final_answer)],
    }
