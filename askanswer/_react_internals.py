"""React 子图（answer ⇄ tools）的工具执行与 HITL 管线。

这些函数不是包的公开接口 —— 一律通过 ``react.build_react_subgraph`` 间接使用。
主推理节点 ``_answer_node``（模型路由 / 上下文预算 / prompt 组装）在
``answering.py``；本文件只保留工具分发与确认执行。

工具路由策略：

* 普通工具调用走 ``langgraph.prebuilt.ToolNode``，由它统一处理并发执行、错误包装、
  以及 ``ToolRuntime`` 注入（这样 ``sql_query`` 等工具能拿到父图的 ``ContextSchema``）。
* 注册时设置 ``confirmation_class`` 的工具（shell / fs_write / external_api_paid）
  会先经过 ``_confirm_plan_node`` 按类规划出“要执行的具体动作”写入 state，再由
  ``_run_with_confirmation`` 通过 ``interrupt()`` 暂停图、交给 CLI 让人类确认。
  每个确认类的规划/闸门/执行逻辑都在 ``confirmations.py``，这里只做通用分发。
"""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from .answering import _emit_tool_telemetry
from .audit import log_event, summarize_args
from .confirmations import get_confirmation_handler
from .registry import get_registry
from .schema import ContextSchema
from .state import SearchState


def _confirm_plan_node(state: SearchState) -> dict:
    """对需要确认的工具调用，按确认类预先规划动作并写入 ``pending_confirmations``。

    通过父图的 checkpointer 持久化下来：之后 ``interrupt()`` + 用户 resume 后
    ``_tools_node`` 直接从 state 里读取已规划好的动作，避免重复调用 LLM
    （否则不仅多花 token，还可能生成不一样的动作导致用户白确认一次）。
    """
    confirmation_classes = get_registry().confirmation_classes()
    plans: dict = dict(state.get("pending_confirmations") or {})
    for tc in state["messages"][-1].tool_calls:
        clazz = confirmation_classes.get(tc["name"])
        handler = get_confirmation_handler(clazz)
        # 只处理已接入执行体的确认类；未接入的类型由 _tools_node 返回友好错误
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


def _tools_node(
    state: SearchState,
    runtime: Runtime[ContextSchema],
) -> dict:
    """工具调用执行节点：把普通工具与需要确认的工具分别派发。"""
    registry = get_registry()
    confirmation_classes = registry.confirmation_classes()
    confirmable_names = {
        name
        for name, cls in confirmation_classes.items()
        if get_confirmation_handler(cls) is not None
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
                    _emit_tool_telemetry(
                        tool_name=tc["name"], duration_ms=duration_ms, error=str(exc)
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
                    _emit_tool_telemetry(tool_name=tc["name"], duration_ms=duration_ms)
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

    # 需要确认的调用一条条走 HITL 流程（内部会 interrupt）
    for tc in confirm_calls:
        out_messages.append(_run_with_confirmation(tc, state))

    # 执行完后清空规划缓存，避免下一轮被旧规划影响（pending_shell 是旧字段，一并清）
    return {"messages": out_messages, "pending_confirmations": {}, "pending_shell": {}}


def _route_from_answer(state: SearchState):
    """answer 节点之后的路由：是否要继续走工具，是否需要 HITL。"""
    if state["step"] != "tool_called":
        return END
    confirmation_classes = get_registry().confirmation_classes()
    tcs = getattr(state["messages"][-1], "tool_calls", None) or []
    # 只要有一条 tool_call 需要（已接入的）确认，就先去 confirm_plan 把动作规划出来
    if any(
        get_confirmation_handler(confirmation_classes.get(tc.get("name"))) is not None
        for tc in tcs
    ):
        return "confirm_plan"
    return "tools"


# ── HITL helpers ─────────────────────────────────────────────────────

def _run_with_confirmation(tool_call: dict, state: SearchState) -> ToolMessage:
    """执行需要人工确认的工具调用：取出预规划动作、安全闸门、interrupt 询问、执行。

    具体的规划/闸门/执行逻辑由确认类 handler（``confirmations.py``）提供，这里
    只负责通用编排与审计（kind 形如 ``shell_approve`` / ``fs_write_reject``）。
    """
    clazz = get_registry().confirmation_classes().get(tool_call["name"]) or "shell"
    handler = get_confirmation_handler(clazz)
    plans = state.get("pending_confirmations") or {}
    # 兼容升级前挂起的旧 checkpoint：老版本只写 pending_shell（必为 shell 类）
    payload = (
        plans.get(tool_call["id"])
        or (state.get("pending_shell") or {}).get(tool_call["id"])
        or {}
    )

    # 弹确认框之前先过安全闸门（规划失败 / 危险命令等），命中直接拦截
    blocked = payload.get("error") or handler.gate(payload)
    if blocked:
        log_event(
            kind=f"{clazz}_reject",
            tool_name=tool_call["name"],
            args_summary=summarize_args(handler.audit_args(payload)),
            error=blocked,
        )
        return ToolMessage(
            content=blocked,
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )

    # interrupt() 抛出后，父图的 stream 会看到 __interrupt__ 通道，CLI 按
    # payload["type"]（confirm_<class>）弹出对应的确认 UI
    decision = interrupt(handler.interrupt_payload(payload))
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
    _emit_tool_telemetry(
        tool_name=tool_call["name"], duration_ms=duration_ms, error=outcome.error
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
