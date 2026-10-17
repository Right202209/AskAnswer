"""act：执行普通工具与已批准的操作（架构文档 6.10.5）。

* 普通工具走 ``langgraph.prebuilt.ToolNode``：统一并发执行、错误包装、``ToolRuntime``
  注入（``sql_query`` 等工具因此能拿到图的 ``ContextSchema``）。
* 确认类工具只按 ask_approval 已记录的决定执行；本节点绝不 interrupt，
  因此恢复时不会重复执行任何副作用（P9）。
"""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from askanswer.graph.context import ContextSchema
from askanswer.graph.nodes.approvals import BLOCKED_KEY as _BLOCKED_KEY
from askanswer.graph.nodes.approvals import DECISION_KEY as _DECISION_KEY
from askanswer.graph.state import SearchState
from askanswer.observability.audit import log_event, summarize_args
from askanswer.runtime import Container


def _act(c: Container, state: SearchState) -> dict:
    """工具调用执行节点：把普通工具与需要确认的工具分别派发。"""
    registry = c.tools
    confirmation_classes = registry.confirmation_classes()
    confirmable_names = {
        name
        for name, cls in confirmation_classes.items()
        if c.confirmations(cls) is not None
    }
    last_msg = state["messages"][-1]
    tool_calls = list(getattr(last_msg, "tool_calls", None) or [])

    # 按是否需要 HITL 确认拆分
    confirm_calls = [tc for tc in tool_calls if tc["name"] in confirmable_names]
    unsupported_confirm_calls = [
        tc
        for tc in tool_calls
        if tc["name"] in confirmation_classes and tc["name"] not in confirmable_names
    ]
    plain_calls = [tc for tc in tool_calls if tc["name"] not in confirmation_classes]

    out_messages: list[ToolMessage] = []

    if plain_calls:
        # 把每个工具调用映射到注册表里的描述符，找不到的视为未知工具
        plain_descriptors = [registry.get(tc["name"]) for tc in plain_calls]
        known_calls = [
            tc for tc, descriptor in zip(plain_calls, plain_descriptors)
            if descriptor is not None
        ]
        plain_tools = [
            descriptor.tool for descriptor in plain_descriptors
            if descriptor is not None
        ]
        unknown = [tc for tc, descriptor in zip(plain_calls, plain_descriptors) if descriptor is None]

        if plain_tools:
            # ToolNode 会自动从 LangGraph 的 contextvar 读取父运行时上下文，
            # 因此工具里 ToolRuntime[ContextSchema] 能直接拿到父图的 ContextSchema。
            tool_node = ToolNode(plain_tools, handle_tool_errors=True)
            # 临时构造一个只含“普通工具调用”的 AIMessage 副本喂给 ToolNode，
            # 避免 ToolNode 把需要确认的工具也跑了
            sub_msg = _clone_message_with_calls(last_msg, known_calls)
            sub_state = {"messages": list(state["messages"][:-1]) + [sub_msg]}
            started = time.monotonic()
            try:
                result = tool_node.invoke(sub_state)
            except Exception as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                # 兜底：把异常包成 ToolMessage 返回，避免整个图崩溃
                for tc in known_calls:
                    log_event(
                        kind="tool_call",
                        tool_name=tc["name"],
                        args_summary=summarize_args(tc.get("args") or {}),
                        duration_ms=duration_ms,
                        error=str(exc),
                    )
                    out_messages.append(
                        ToolMessage(
                            content=f"工具 {tc['name']} 执行失败：{exc}",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                    )
            else:
                duration_ms = int((time.monotonic() - started) * 1000)
                result_messages = result.get("messages", [])
                by_call_id = {
                    getattr(message, "tool_call_id", None): message
                    for message in result_messages
                }
                for tc in known_calls:
                    message = by_call_id.get(tc.get("id"))
                    content = getattr(message, "content", "") if message else ""
                    log_event(
                        kind="tool_call",
                        tool_name=tc["name"],
                        args_summary=summarize_args(tc.get("args") or {}),
                        result_size=len(str(content)),
                        duration_ms=duration_ms,
                    )
                out_messages.extend(result_messages)

        # 未知工具：返回“未注册”消息，让 LLM 下一轮换工具
        for tc in unknown:
            log_event(
                kind="tool_call",
                tool_name=tc["name"],
                args_summary=summarize_args(tc.get("args") or {}),
                error="unknown tool",
            )
            out_messages.append(
                ToolMessage(
                    content=f"未知工具：{tc['name']}",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )

    for tc in unsupported_confirm_calls:
        log_event(
            kind="tool_call",
            tool_name=tc["name"],
            args_summary=summarize_args(tc.get("args") or {}),
            error=f"unsupported confirmation class: {confirmation_classes[tc['name']]}",
        )
        out_messages.append(
            ToolMessage(
                content=(
                    f"工具 {tc['name']} 需要 {confirmation_classes[tc['name']]} 确认，"
                    "但当前执行层尚未接入该确认类型。"
                ),
                tool_call_id=tc["id"],
                name=tc["name"],
            )
        )

    # 需要确认的调用：ask_approval 节点已逐条拿到用户决定并写入 state，
    # 这里只按决定执行（apply 内部会二次校验），本节点绝不 interrupt。
    for tc in confirm_calls:
        out_messages.append(_apply_confirmation(c, tc, state))

    # 执行完后清空规划缓存，避免下一轮被旧规划影响
    return {"messages": out_messages, "pending_confirmations": {}}


def _apply_confirmation(c: Container, tool_call: dict, state: SearchState) -> ToolMessage:
    """按 ask_approval 记录的决定执行一个确认类工具调用，并写审计。"""
    clazz = c.tools.confirmation_classes().get(tool_call["name"]) or "shell"
    handler = c.confirmations(clazz)
    plans = state.get("pending_confirmations") or {}
    payload = dict(plans.get(tool_call["id"]) or {})
    blocked = payload.pop(_BLOCKED_KEY, None)
    if blocked or _DECISION_KEY not in payload:
        reason = blocked or payload.get("error") or "未获得用户确认，已跳过执行"
        log_event(
            kind=f"{clazz}_reject",
            tool_name=tool_call["name"],
            args_summary=summarize_args(handler.audit_args(payload)),
            error=reason,
        )
        return ToolMessage(
            content=reason,
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
    decision = payload.pop(_DECISION_KEY)
    # apply 内部会重新过安全闸门（用户可能编辑过内容），然后才真正执行
    started = time.monotonic()
    outcome = handler.apply(payload, decision, tool_call)
    duration_ms = int((time.monotonic() - started) * 1000)
    log_event(
        kind=f"{clazz}_approve" if outcome.approved else f"{clazz}_reject",
        tool_name=tool_call["name"],
        args_summary=summarize_args(outcome.audit_args or handler.audit_args(payload)),
        result_size=len(outcome.content) if outcome.approved else None,
        duration_ms=duration_ms,
        error=outcome.error,
    )
    return ToolMessage(
        content=outcome.content,
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )


def _clone_message_with_calls(message: AIMessage, tool_calls: list[dict]) -> AIMessage:
    """复制一份 AIMessage 但只保留指定的 ``tool_calls`` 子集。

    用途：把仅含“普通工具调用”的副本喂给 ToolNode，原始 AIMessage 仍保留在
    state 的消息历史里，不会被 ToolNode 改写。
    """
    try:
        # pydantic v2 路径
        return message.model_copy(update={"tool_calls": list(tool_calls)})
    except AttributeError:  # pragma: no cover — 兼容旧版 pydantic v1
        clone = message.copy()
        clone.tool_calls = list(tool_calls)
        return clone


def make_act_node(c: Container):
    def act(state: SearchState, runtime: Runtime[ContextSchema]) -> dict:
        # runtime 参数保留：让 LangGraph 把 ContextSchema 放进 contextvar，
        # ToolNode 内的 ToolRuntime 注入依赖它。
        return _act(c, state)

    return act
