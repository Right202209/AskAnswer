"""Internal nodes for the react (answer ⇄ tools) subgraph.

These functions are not part of the package's public surface — call them via
``react.build_react_subgraph`` instead.

Tool routing here:

* Plain tool calls go through ``langgraph.prebuilt.ToolNode``, which handles
  parallel execution, error wrapping, and ``ToolRuntime`` injection (so e.g.
  ``sql_query`` sees the parent runtime ``ContextSchema``).
* Tools whose registry entry sets ``requires_confirmation=True`` (today: only
  ``gen_shell_commands_run``) are pre-planned by ``_shell_plan_node`` and
  dispatched through ``_run_with_confirmation``, which raises ``interrupt()``
  so the CLI can prompt the human.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from .load import model
from .nodes import _local_intent
from .registry import get_registry
from .schema import ContextSchema
from .state import SearchState
from .tools import check_dangerous, execute_shell_command, gen_shell_command_spec


def _reclassify_intent(state: SearchState) -> str | None:
    """Re-derive intent from the latest human turn so mid-conversation topic
    shifts (e.g. chat → SQL) rebind the tool bundle on the next iteration.

    Returns the new intent string, or ``None`` if the latest message isn't a
    fresh human turn (we don't flap intent while a tool call is in flight) or
    the local classifier is uncertain.
    """
    if state.get("step") == "retry_search":
        # Sorcery's retry HumanMessage is synthetic, not a fresh user turn —
        # the original intent already settled in ``understand_query_node`` and
        # the retry path only fires for search anyway.
        return None
    messages = state.get("messages") or []
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, HumanMessage):
        return None
    fields = _local_intent(getattr(last, "content", "") or "")
    if fields is None:
        return None
    return fields["intent"]


def _answer_node(state: SearchState) -> dict:
    new_intent = _reclassify_intent(state)
    intent = new_intent or state.get("intent", "search")
    search_results = state.get("search_results", "")

    if intent == "chat":
        context_line = "（这是闲聊或常识类问题，不需要搜索结果；可直接回答或调用合适的工具。）"
    elif intent == "sql":
        context_line = "（这是数据库/SQL 类问题；如需查询数据，调用 sql_query 工具。）"
    elif intent == "file_read":
        file_path = state.get("file_path") or ""
        if file_path:
            context_line = f"（这是读文件请求，请调用 read_file 工具读取 `{file_path}` 后再作答。）"
        else:
            context_line = "（这是读文件请求，请调用 read_file 工具读取目标文件后再作答。）"
    elif not search_results:
        context_line = "（如需联网信息请调用 tavily_search 工具，否则基于已有知识回答。）"
    else:
        context_line = f"以下是搜索结果，可作为参考：\n{search_results}"

    bundle_tools = get_registry().list(bundle=intent)
    tool_names = ", ".join(t.name for t in bundle_tools) or "(无)"

    system_prompt = (
        "你可以调用工具来协助用户。\n"
        f"当前可用工具：{tool_names}。\n"
        "若用户需要相应信息，直接调用对应工具；否则结合上下文直接回答。\n\n"
        f"用户查询解析：{state.get('user_query', '')}\n"
        f"{context_line}"
    )

    bound = model.bind_tools(bundle_tools) if bundle_tools else model
    msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = bound.invoke(msgs)

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        out: dict = {"step": "tool_called", "messages": [response]}
        if new_intent and new_intent != state.get("intent"):
            out["intent"] = new_intent
        return out
    out = {
        "final_answer": response.content,
        "step": "completed",
        "messages": [response],
    }
    if new_intent and new_intent != state.get("intent"):
        out["intent"] = new_intent
    return out


def _shell_plan_node(state: SearchState) -> dict:
    """Pre-generate shell commands for any confirmation-required tool calls.

    Persisted via the parent checkpointer so that, after ``interrupt()`` and
    the user's resume, ``_tools_node`` can read the plan from state instead of
    re-invoking the LLM (which would burn tokens and might give a different
    command).
    """
    confirmation_names = get_registry().confirmation_names()
    plans: dict = dict(state.get("pending_shell") or {})
    for tc in state["messages"][-1].tool_calls:
        if tc["name"] not in confirmation_names:
            continue
        if tc["id"] in plans:
            continue
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
    registry = get_registry()
    confirmation_names = registry.confirmation_names()
    last_msg = state["messages"][-1]
    tool_calls = list(getattr(last_msg, "tool_calls", None) or [])

    confirm_calls = [tc for tc in tool_calls if tc["name"] in confirmation_names]
    plain_calls = [tc for tc in tool_calls if tc["name"] not in confirmation_names]

    out_messages: list[ToolMessage] = []

    if plain_calls:
        plain_descriptors = [registry.get(tc["name"]) for tc in plain_calls]
        plain_tools = [d.tool for d in plain_descriptors if d is not None]
        unknown = [tc for tc, d in zip(plain_calls, plain_descriptors) if d is None]

        if plain_tools:
            # ToolNode reads the parent runtime from the LangGraph contextvar
            # set by the surrounding node execution, so ``ToolRuntime`` in
            # tools (e.g. ``sql_query``) gets the parent ``ContextSchema``.
            tool_node = ToolNode(plain_tools, handle_tool_errors=True)
            sub_msg = _clone_message_with_calls(last_msg, plain_calls)
            sub_state = {"messages": list(state["messages"][:-1]) + [sub_msg]}
            try:
                result = tool_node.invoke(sub_state)
            except Exception as exc:
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

        for tc in unknown:
            out_messages.append(
                ToolMessage(
                    content=f"未知工具：{tc['name']}",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )

    for tc in confirm_calls:
        out_messages.append(_run_with_confirmation(tc, state))

    return {"messages": out_messages, "pending_shell": {}}


def _route_from_answer(state: SearchState):
    if state["step"] != "tool_called":
        return END
    confirmation_names = get_registry().confirmation_names()
    tcs = getattr(state["messages"][-1], "tool_calls", None) or []
    if any(tc.get("name") in confirmation_names for tc in tcs):
        return "shell_plan"
    return "tools"


# ── HITL helpers ─────────────────────────────────────────────────────

def _run_with_confirmation(tool_call: dict, state: SearchState) -> ToolMessage:
    plan = (state.get("pending_shell") or {}).get(tool_call["id"]) or {}
    command = plan.get("command") or ""
    explanation = plan.get("explanation") or ""
    if not command:
        return ToolMessage(
            content=explanation or "未能生成有效的 shell 命令",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )

    danger = check_dangerous(command)
    if danger:
        return ToolMessage(
            content=f"已拦截高风险命令（{danger}）：{command}",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )

    decision = interrupt(
        {
            "type": "confirm_shell",
            "command": command,
            "explanation": explanation,
            "instruction": plan.get("instruction", ""),
        }
    )
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
    return ToolMessage(
        content=execute_shell_command(approved_command),
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )


def _parse_decision(decision: Any, fallback_command: str) -> tuple[bool, str]:
    if decision is True:
        return True, fallback_command
    if isinstance(decision, dict):
        approve = decision.get("approve")
        if approve is None:
            approve = decision.get("value")
        cmd = decision.get("command") or fallback_command
        if isinstance(approve, bool):
            return approve, cmd
        return _truthy(approve), cmd
    return _truthy(decision), fallback_command


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"y", "yes", "true", "1", "approve"}


def _clone_message_with_calls(message: AIMessage, tool_calls: list[dict]) -> AIMessage:
    """Return a copy of ``message`` whose ``tool_calls`` is filtered to ``tool_calls``.

    Used to feed only the plain (non-confirmation) tool_calls into ``ToolNode``
    while keeping the original AIMessage in the state's message history.
    """
    try:
        return message.model_copy(update={"tool_calls": list(tool_calls)})
    except AttributeError:  # pragma: no cover — older pydantic v1 path
        clone = message.copy()
        clone.tool_calls = list(tool_calls)
        return clone
