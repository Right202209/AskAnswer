# 父图前置节点：意图分类（understand）和事后评估（sorcery）。
# 设计原则：先用本地正则/关键词做高置信度分类，避免每次都调用 LLM；
# 仅当本地启发不确定时才回退到 LLM 来判定 intent。
#
# 节点由工厂 ``make_xxx_node(container)`` 生成（架构文档 P2 / 3.2）：依赖从
# Container 注入，节点体内不 import 模块级单例。
from __future__ import annotations

from langchain_core.messages import SystemMessage

from .audit import log_event, run_usage_so_far
from .intents.base import IntentClassification
from .routing import ROLE_CLASSIFY
from .runtime import Container
from .state import (
    PHASE_COMPLETED,
    PHASE_RETRY,
    PHASE_UNDERSTOOD,
    SearchState,
    with_slots,
)


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


def make_understand_node(container: Container):
    def understand(state: SearchState) -> dict:
        """父图第一个节点：把用户最新消息分类为某个 intent，并把参数写进 slots。"""
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

    return understand


def _run_budget_stop_reason(budget: int | None) -> str | None:
    """成本闸门：本轮 token 消耗超出预算时返回停止原因；未配置/未超出返回 None。

    读取的是 audit 在本次 run 内累计的 llm_call 用量（内存缓冲，不查库），
    因此闸门本身零额外开销；预算未配置时完全不启用（非回归默认）。
    """
    if budget is None:
        return None
    used_input, used_output = run_usage_so_far()
    used = used_input + used_output
    if used < budget:
        return None
    return f"本轮已消耗约 {used} tokens ≥ 预算 {budget}，跳过质量重试直接收尾"


def make_sorcery_node(container: Container):
    def sorcery(state: SearchState) -> dict:
        """父图第三个节点：把答案质量评估委托给当前 intent handler。"""
        handler = container.intents.get(state.get("intent", "search"))

        retry_count = state.get("retry_count", 0)
        if retry_count >= handler.max_retries:
            return _finalize(state)

        # 成本闸门先于质量评估：预算耗尽时连评估这次 LLM 调用也省掉
        budget_stop = _run_budget_stop_reason(container.settings.run_token_budget)
        if budget_stop:
            log_event(
                kind="budget_stop",
                intent=state.get("intent"),
                args_summary=budget_stop,
            )
            return _finalize(state)

        result = handler.evaluate(state)
        if result.decision == "pass":
            return _finalize(state)

        retry_directive = result.retry_directive or {}
        out = {
            "retry_count": retry_count + 1,
            "retry_directive": retry_directive,
            "step": PHASE_RETRY,
        }
        # 重试指令里的 slots（如新的 search_query）写回 state.slots
        slot_updates = {
            k: v for k, v in (retry_directive.get("slots") or {}).items() if v
        }
        if slot_updates:
            out["slots"] = with_slots(state, **slot_updates)
        return out

    return sorcery


def _finalize(state: SearchState) -> dict:
    """sorcery 评估通过 / 重试预算耗尽后的统一收尾：清空 retry_directive 并打 step。

    最终答案已由 answer 节点作为 AIMessage 写入历史，这里不再重复追加（P6）。
    """
    final_answer = state.get("final_answer", "")
    return {
        "final_answer": final_answer,
        "step": PHASE_COMPLETED,
        "retry_directive": {},
    }
