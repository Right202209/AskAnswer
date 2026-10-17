"""review：按 intent 的策略评估答案、检查本轮 token 预算、决定是否重试（6.10.6）。"""

from __future__ import annotations

from askanswer.graph.state import PHASE_COMPLETED, PHASE_RETRY, SearchState, with_slots
from askanswer.intents.base import run_evaluate
from askanswer.observability.audit import log_event, run_usage_so_far
from askanswer.runtime import Container


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


def make_review_node(container: Container):
    def review(state: SearchState) -> dict:
        """把答案质量评估委托给当前 intent handler；需要重试时回到 reason。"""
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

        result = run_evaluate(handler, state, container.models)
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

    return review


def _finalize(state: SearchState) -> dict:
    """review 评估通过 / 重试预算耗尽后的统一收尾：清空 retry_directive 并打 step。

    最终答案已由 reason 节点作为 AIMessage 写入历史，这里不再重复追加（P6）。
    """
    final_answer = state.get("final_answer", "")
    return {
        "final_answer": final_answer,
        "step": PHASE_COMPLETED,
        "retry_directive": {},
    }
