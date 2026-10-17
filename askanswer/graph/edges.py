"""主图的条件路由函数（架构文档 6.10.7）。"""

from __future__ import annotations

from langgraph.graph import END

from askanswer.graph.nodes.approvals import undecided_ids
from askanswer.graph.state import PHASE_RETRY, PHASE_TOOL_CALLED, SearchState
from askanswer.runtime import Container


def make_route_after_reason(c: Container):
    def route_after_reason(state: SearchState) -> str:
        """reason 之后：无工具调用 → review；有需审批的 → plan_approvals；否则 → act。"""
        if state.get("step") != PHASE_TOOL_CALLED:
            return "review"
        confirmation_classes = c.tools.confirmation_classes()
        tcs = getattr(state["messages"][-1], "tool_calls", None) or []
        if any(
            c.confirmations(confirmation_classes.get(tc.get("name"))) is not None
            for tc in tcs
        ):
            return "plan_approvals"
        return "act"

    return route_after_reason


def route_after_approval(state: SearchState) -> str:
    """还有未决定的审批项就继续询问，全部决定后进入 act。"""
    return "ask_approval" if undecided_ids(state) else "act"


def route_after_review(state: SearchState):
    """review 判定需要重试时回到 reason，否则结束本轮。"""
    return "reason" if state.get("step") == PHASE_RETRY else END
