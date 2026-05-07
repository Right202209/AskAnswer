"""React 子图（answer ⇄ tools）的内部节点实现。

这些函数不是包的公开接口 —— 一律通过 ``react.build_react_subgraph`` 间接使用。

工具路由策略：

* 普通工具调用走 ``langgraph.prebuilt.ToolNode``，由它统一处理并发执行、错误包装、
  以及 ``ToolRuntime`` 注入（这样 ``sql_query`` 等工具能拿到父图的 ``ContextSchema``）。
* 注册时设置 ``confirmation_class="shell"`` 的工具（目前只有 ``gen_shell_commands_run``）
  会先经过 ``_shell_plan_node`` 预先生成命令并写入 state，再由 ``_run_with_confirmation``
  通过 ``interrupt()`` 暂停图、把命令交给 CLI 让人类确认。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from .intents import get_intent_registry
from .load import model
from .registry import get_registry
from .schema import ContextSchema
from .state import SearchState
from .tools import check_dangerous, execute_shell_command, gen_shell_command_spec


def _reclassify_intent(state: SearchState) -> str | None:
    """根据“最新一条用户消息”重新判定 intent，让会话中途切换主题时也能换工具集。

    返回新的 intent 字符串；返回 ``None`` 表示这次不需要切换（例如最新一条不是
    新的真人输入，或本地分类器拿不准）。
    """
    if state.get("step") == "retry_search":
        # sorcery 触发的“重新搜索” HumanMessage 是合成的，不是真正的新一轮提问，
        # 原始 intent 已在 understand 阶段确定，且这里只可能是 search，跳过避免反转。
        return None
    messages = state.get("messages") or []
    if not messages:
        return None
    last = messages[-1]
    # 工具调用回填的消息不是 HumanMessage，跳过避免在工具链中途切换 intent
    if not isinstance(last, HumanMessage):
        return None
    fields = get_intent_registry().classify_local(getattr(last, "content", "") or "")
    if fields is None:
        return None
    return fields.intent


def _answer_node(state: SearchState) -> dict:
    """react 主推理节点：根据 intent 拼 system prompt、绑定对应工具、调用 LLM。"""
    # 中途话题切换的兜底：用户突然从 chat 转到 sql 等，需要切换工具集
    new_intent = _reclassify_intent(state)
    intent = new_intent or state.get("intent", "search")
    handler = get_intent_registry().get(intent)
    context_line = handler.prompt_hint(state)
    retry_directive = dict(state.get("retry_directive") or {})
    if retry_directive:
        directive = retry_directive.get("instruction") or retry_directive
        context_line = f"{context_line}\n\n上一次回答不够，请按以下指引重试：{directive}"

    # 从注册表按 handler 的 tag 集合取工具，新增 intent 不必改 registry 常量。
    bundle_tools = get_registry().list(tags=handler.bundle_tags)
    tool_names = ", ".join(t.name for t in bundle_tools) or "(无)"

    system_prompt = (
        "你可以调用工具来协助用户。\n"
        f"当前可用工具：{tool_names}。\n"
        "若用户需要相应信息，直接调用对应工具；否则结合上下文直接回答。\n\n"
        f"用户查询解析：{state.get('user_query', '')}\n"
        f"{context_line}"
    )

    # 没有可用工具时退化为纯对话；否则把工具绑定后再调用
    bound = model.bind_tools(bundle_tools) if bundle_tools else model
    msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = bound.invoke(msgs)

    # LLM 是否要求继续调用工具
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        out: dict = {"step": "tool_called", "messages": [response]}
        if retry_directive:
            out["retry_directive"] = {}
        # 中途切换 intent 时，把新 intent 写回 state，下一轮 tool 选择就会变化
        if new_intent and new_intent != state.get("intent"):
            out["intent"] = new_intent
        return out
    # LLM 直接给出最终答案，准备进入 sorcery 节点评估
    out = {
        "final_answer": response.content,
        "step": "completed",
        "messages": [response],
    }
    if retry_directive:
        out["retry_directive"] = {}
    if new_intent and new_intent != state.get("intent"):
        out["intent"] = new_intent
    return out


def _shell_plan_node(state: SearchState) -> dict:
    """对需要确认的工具调用，预先生成 shell 命令并写入 ``pending_shell``。

    通过父图的 checkpointer 持久化下来：之后 ``interrupt()`` + 用户 resume 后
    ``_tools_node`` 直接从 state 里读取已规划好的命令，避免重复调用 LLM
    （否则不仅多花 token，还可能生成不一样的命令导致用户白确认一次）。
    """
    confirmation_classes = get_registry().confirmation_classes()
    shell_names = {name for name, cls in confirmation_classes.items() if cls == "shell"}
    plans: dict = dict(state.get("pending_shell") or {})
    for tc in state["messages"][-1].tool_calls:
        # 只处理 shell 类人工确认；其它确认类型不能复用 shell 规划器。
        if tc["name"] not in shell_names:
            continue
        # 已经规划过就不重复生成（resume 后再次进入此节点时跳过）
        if tc["id"] in plans:
            continue
        # 工具入参里取自然语言指令；instruction 优先，input 兜底
        instruction = (
            (tc.get("args") or {}).get("instruction")
            or (tc.get("args") or {}).get("input")
            or ""
        ).strip()
        if not instruction:
            plans[tc["id"]] = {
                "command": "",
                "explanation": "未提供 shell 指令",
                "instruction": "",
            }
            continue
        try:
            command, explanation = gen_shell_command_spec(instruction)
        except Exception as exc:
            # 生成失败也写入 plans，让后续节点据此返回友好错误而不是抛异常
            plans[tc["id"]] = {
                "command": "",
                "explanation": f"生成 shell 命令失败：{exc}",
                "instruction": instruction,
            }
            continue
        plans[tc["id"]] = {
            "command": command,
            "explanation": explanation,
            "instruction": instruction,
        }
    return {"pending_shell": plans}


def _tools_node(
    state: SearchState,
    runtime: Runtime[ContextSchema],
) -> dict:
    """工具调用执行节点：把普通工具与需要确认的工具分别派发。"""
    registry = get_registry()
    confirmation_classes = registry.confirmation_classes()
    shell_names = {name for name, cls in confirmation_classes.items() if cls == "shell"}
    last_msg = state["messages"][-1]
    tool_calls = list(getattr(last_msg, "tool_calls", None) or [])

    # 按是否需要 HITL 确认拆分
    confirm_calls = [tc for tc in tool_calls if tc["name"] in shell_names]
    unsupported_confirm_calls = [
        tc
        for tc in tool_calls
        if tc["name"] in confirmation_classes and tc["name"] not in shell_names
    ]
    plain_calls = [tc for tc in tool_calls if tc["name"] not in confirmation_classes]

    out_messages: list[ToolMessage] = []

    if plain_calls:
        # 把每个工具调用映射到注册表里的描述符，找不到的视为未知工具
        plain_descriptors = [registry.get(tc["name"]) for tc in plain_calls]
        plain_tools = [d.tool for d in plain_descriptors if d is not None]
        unknown = [tc for tc, d in zip(plain_calls, plain_descriptors) if d is None]

        if plain_tools:
            # ToolNode 会自动从 LangGraph 的 contextvar 读取父运行时上下文，
            # 因此工具里 ToolRuntime[ContextSchema] 能直接拿到父图的 ContextSchema。
            tool_node = ToolNode(plain_tools, handle_tool_errors=True)
            # 临时构造一个只含“普通工具调用”的 AIMessage 副本喂给 ToolNode，
            # 避免 ToolNode 把需要确认的工具也跑了
            sub_msg = _clone_message_with_calls(last_msg, plain_calls)
            sub_state = {"messages": list(state["messages"][:-1]) + [sub_msg]}
            try:
                result = tool_node.invoke(sub_state)
            except Exception as exc:
                # 兜底：把异常包成 ToolMessage 返回，避免整个图崩溃
                for tc in plain_calls:
                    out_messages.append(
                        ToolMessage(
                            content=f"工具 {tc['name']} 执行失败：{exc}",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                    )
            else:
                out_messages.extend(result.get("messages", []))

        # 未知工具：返回“未注册”消息，让 LLM 下一轮换工具
        for tc in unknown:
            out_messages.append(
                ToolMessage(
                    content=f"未知工具：{tc['name']}",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )

    for tc in unsupported_confirm_calls:
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

    # 执行完后清空 pending_shell，避免下一轮被旧规划影响
    return {"messages": out_messages, "pending_shell": {}}


def _route_from_answer(state: SearchState):
    """answer 节点之后的路由：是否要继续走工具，是否需要 HITL。"""
    if state["step"] != "tool_called":
        return END
    confirmation_classes = get_registry().confirmation_classes()
    shell_names = {name for name, cls in confirmation_classes.items() if cls == "shell"}
    tcs = getattr(state["messages"][-1], "tool_calls", None) or []
    # 只要有一条 tool_call 需要确认，就先去 shell_plan 把命令规划出来
    if any(tc.get("name") in shell_names for tc in tcs):
        return "shell_plan"
    return "tools"


# ── HITL helpers ─────────────────────────────────────────────────────

def _run_with_confirmation(tool_call: dict, state: SearchState) -> ToolMessage:
    """执行需要人工确认的工具调用：取出预生成命令、危险检查、interrupt 询问、执行。"""
    plan = (state.get("pending_shell") or {}).get(tool_call["id"]) or {}
    command = plan.get("command") or ""
    explanation = plan.get("explanation") or ""
    # 没规划出有效命令直接返回错误信息，不往后走
    if not command:
        return ToolMessage(
            content=explanation or "未能生成有效的 shell 命令",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )

    # 用户确认前先做一次危险命令检查，避免诱导用户点 y 后造成损失
    danger = check_dangerous(command)
    if danger:
        return ToolMessage(
            content=f"已拦截高风险命令（{danger}）：{command}",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )

    # interrupt() 抛出后，父图的 stream 会看到 __interrupt__ 通道，CLI 弹出确认 UI
    decision = interrupt(
        {
            "type": "confirm_shell",
            "command": command,
            "explanation": explanation,
            "instruction": plan.get("instruction", ""),
        }
    )
    # 用户可能修改了命令，需要重新做危险检查
    approved, approved_command = _parse_decision(decision, fallback_command=command)
    if not approved:
        return ToolMessage(
            content=f"已取消执行：{approved_command}",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
    danger = check_dangerous(approved_command)
    if danger:
        return ToolMessage(
            content=f"已拦截高风险命令（{danger}）：{approved_command}",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
    # 真正执行命令；输出包装为 ToolMessage 回填到对话
    return ToolMessage(
        content=execute_shell_command(approved_command),
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )


def _parse_decision(decision: Any, fallback_command: str) -> tuple[bool, str]:
    """把 CLI 的 resume 值统一解析成 (approved, command)。"""
    # 直接传 True：批准，沿用旧命令
    if decision is True:
        return True, fallback_command
    if isinstance(decision, dict):
        # 字典形式：兼容 approve / value 两个键名
        approve = decision.get("approve")
        if approve is None:
            approve = decision.get("value")
        cmd = decision.get("command") or fallback_command
        if isinstance(approve, bool):
            return approve, cmd
        return _truthy(approve), cmd
    return _truthy(decision), fallback_command


def _truthy(value: Any) -> bool:
    """把字符串/None 等输入转换成布尔，用来兼容人类敲的 y/yes/1 等。"""
    if value is None:
        return False
    return str(value).strip().lower() in {"y", "yes", "true", "1", "approve"}


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
