"""审批：plan_approvals → ask_approval（架构文档 6.10.4）。

旧实现把「执行普通工具」和「interrupt 询问」放在同一个 tools 节点里：LangGraph
恢复时会从节点开头重跑，导致普通工具被执行两次、已批准的动作被重放（P9）。
现在拆成三段，本文件是前两段：

* ``plan_approvals``  按确认类规划要执行的具体动作，写 pending_confirmations（无副作用）
* ``ask_approval``    每次只询问一个尚无决定的项，把决定写回 state；被安全闸门拦截的项
                      直接记为 blocked，不打扰用户（无副作用，恢复时重跑安全）
* ``act``（act.py）   执行普通工具 + 按已记录的决定 apply（绝不 interrupt）
"""

from __future__ import annotations

from langgraph.types import interrupt

from askanswer.graph.state import SearchState
from askanswer.runtime import Container

DECISION_KEY = "decision"
BLOCKED_KEY = "blocked"
_DECISION_KEY = DECISION_KEY
_BLOCKED_KEY = BLOCKED_KEY


def _plan_approvals(c: Container, state: SearchState) -> dict:
    """对需要确认的工具调用，按确认类预先规划动作并写入 ``pending_confirmations``。

    通过父图的 checkpointer 持久化下来：之后 ``interrupt()`` + 用户 resume 后
    ``act 节点`` 直接从 state 里读取已规划好的动作，避免重复调用 LLM
    （否则不仅多花 token，还可能生成不一样的动作导致用户白确认一次）。
    """
    confirmation_classes = c.tools.confirmation_classes()
    plans: dict = dict(state.get("pending_confirmations") or {})
    for tc in state["messages"][-1].tool_calls:
        clazz = confirmation_classes.get(tc["name"])
        handler = c.confirmations(clazz)
        # 只处理已接入执行体的确认类；未接入的类型由 act 节点 返回友好错误
        if handler is None:
            continue
        # 已经规划过就不重复生成（resume 后再次进入此节点时跳过）
        if tc["id"] in plans:
            continue
        try:
            payload = handler.plan(tc)
        except Exception as exc:
            # 规划失败也写入 plans，让后续节点据此返回友好错误而不是抛异常
            payload = {"error": f"规划确认内容失败：{exc}"}
        plans[tc["id"]] = {"class": clazz, **payload}
    return {"pending_confirmations": plans}


def undecided_ids(state: SearchState) -> list[str]:
    """按 tool_call 顺序列出尚未拿到用户决定的确认项 id。"""
    plans = state.get("pending_confirmations") or {}
    last = state["messages"][-1]
    ids: list[str] = []
    for tc in getattr(last, "tool_calls", None) or []:
        plan = plans.get(tc.get("id"))
        if plan is None:
            continue
        if _DECISION_KEY in plan or _BLOCKED_KEY in plan:
            continue
        ids.append(tc["id"])
    return ids


def _ask_approval(c: Container, state: SearchState) -> dict:
    """一次只处理一个待确认项：先过安全闸门，未拦截才 interrupt 询问用户。

    本节点只读写 state，没有任何外部副作用 —— 恢复时重跑是安全的。
    """
    pending = undecided_ids(state)
    if not pending:
        return {}
    call_id = pending[0]
    plans = dict(state.get("pending_confirmations") or {})
    plan = dict(plans[call_id])
    handler = c.confirmations(plan.get("class"))
    if handler is None:
        plan[_BLOCKED_KEY] = f"未接入的确认类型：{plan.get('class')}"
    else:
        blocked = plan.get("error") or handler.gate(plan)
        if blocked:
            plan[_BLOCKED_KEY] = blocked
        else:
            # interrupt() 抛出后，父图 stream 看到 __interrupt__，CLI 按
            # payload["type"]（confirm_<class>）弹出对应的确认 UI
            plan[_DECISION_KEY] = interrupt(handler.interrupt_payload(plan))
    plans[call_id] = plan
    return {"pending_confirmations": plans}


def make_plan_approvals_node(c: Container):
    def plan_approvals(state: SearchState) -> dict:
        return _plan_approvals(c, state)

    return plan_approvals


def make_ask_approval_node(c: Container):
    def ask_approval(state: SearchState) -> dict:
        return _ask_approval(c, state)

    return ask_approval
