"""route：选 intent、提取参数、初始化本轮状态（架构文档 6.10.1）。

先用本地正则/关键词做高置信度分类，避免每次都调 LLM；拿不准才用 classify 角色模型。
节点由工厂生成，依赖从 Container 注入。
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from askanswer.graph.state import (
    PHASE_UNDERSTOOD,
    SearchState,
)
from askanswer.intents.base import IntentClassification
from askanswer.llm.routing import ROLE_CLASSIFY
from askanswer.runtime import Container


def classify_intent(container: Container, user_message: str) -> dict:
    """本地分类器优先；拿不准再调 LLM；LLM 也失败就强制兜底，避免流程卡死。"""
    intents = container.intents
    fields = intents.classify_local(user_message)
    if fields is not None:
        return fields.model_dump()
    try:
        prompt = intents.classification_prompt(user_message)
        classifier = container.models(ROLE_CLASSIFY).with_structured_output(
            IntentClassification
        )
        response = classifier.invoke([SystemMessage(content=prompt)])
        return intents.normalize(response.model_dump(), user_message).model_dump()
    except Exception:
        return intents.classify_local(user_message, fallback=True).model_dump()


def make_route_node(container: Container):
    def route(state: SearchState) -> dict:
        """主图第一个节点：选 intent、提取参数写进 slots、初始化本轮状态。"""
        user_message = state["messages"][-1].content
        fields = classify_intent(container, user_message)
        slots = IntentClassification.model_validate(fields).to_slots()
        # 注意：节点返回的是“部分状态”字典，会被 LangGraph 合并到 SearchState
        return {
            "user_query": fields["understanding"],
            # intent 专用参数统一进 slots（P4）；每轮新问题都整体替换，不继承上一轮
            "slots": slots,
            "intent": fields["intent"],
            "retry_count": 0,
            "retry_directive": {},
            "step": PHASE_UNDERSTOOD,
            # 不再往 messages 里追加进度提示（P6）：进度由 CLI 按 intent 等字段渲染。
        }

    return route
