from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import interrupt

from .load import model
from .mcp import get_manager as _mcp_manager
from .state import SearchState
from .tools import (
    check_dangerous,
    execute_shell_command,
    gen_shell_command_spec,
    tools,
    tools_by_name,
)


SHELL_TOOL_NAME = "gen_shell_commands_run"


def _mcp_tool_specs() -> list[dict]:
    """Convert registered MCP tools into OpenAI-style function-call dict specs."""
    specs: list[dict] = []
    try:
        mcp_tools = _mcp_manager().list_tools()
    except Exception:
        return specs

    for t in mcp_tools:
        schema = t.get("input_schema")
        if not isinstance(schema, dict) or not schema:
            schema = {"type": "object", "properties": {}}
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description") or t["name"],
                    "parameters": schema,
                },
            }
        )
    return specs


def _mcp_tool_names() -> set[str]:
    try:
        return {t["name"] for t in _mcp_manager().list_tools()}
    except Exception:
        return set()


def _answer_node(state: SearchState) -> dict:
    intent = state.get("intent", "search")
    search_results = state.get("search_results", "")

    if intent == "chat":
        context_line = "（这是闲聊或常识类问题，不需要搜索结果；可直接回答或调用合适的工具。）"
    elif state.get("step") == "search_failed":
        context_line = "（搜索 API 暂不可用，请基于已有知识或调用工具回答。）"
    elif not search_results:
        context_line = "（没有可用的搜索结果，请基于已有知识或调用工具回答。）"
    else:
        context_line = f"以下是搜索结果，可作为参考：\n{search_results}"

    mcp_specs = _mcp_tool_specs()
    mcp_line = ""
    if mcp_specs:
        names = ", ".join(spec["function"]["name"] for spec in mcp_specs)
        mcp_line = f"\n额外的 MCP 工具可直接按名称调用：{names}。"

    system_prompt = (
        "你可以调用工具来协助用户。\n"
        "可用工具：read_file（读取本地 .txt/.md/.json/.csv/.xlsx）、"
        "check_weather、get_current_time、calculate、convert_currency、lookup_ip。\n"
        f"若用户需要相应信息，直接调用对应工具；否则结合上下文直接回答。{mcp_line}\n\n"
        f"用户查询解析：{state.get('user_query', '')}\n"
        f"{context_line}"
    )

    bound = model.bind_tools(tools + mcp_specs) if mcp_specs else model.bind_tools(tools)
    msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = bound.invoke(msgs)

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        return {
            "step": "tool_called",
            "messages": [response],
        }

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [response],
    }


def _shell_plan_node(state: SearchState) -> dict:
    """为本轮 tool_calls 中的 shell 调用预生成命令，写入 state。"""
    plans: dict = dict(state.get("pending_shell") or {})
    for tc in state["messages"][-1].tool_calls:
        if tc["name"] != SHELL_TOOL_NAME:
            continue
        if tc["id"] in plans:
            continue
        instruction = ((tc.get("args") or {}).get("instruction")
                       or (tc.get("args") or {}).get("input")
                       or "").strip()
        if not instruction:
            plans[tc["id"]] = {"command": "", "explanation": "未提供 shell 指令", "instruction": ""}
            continue
        try:
            command, explanation = gen_shell_command_spec(instruction)
        except Exception as exc:
            plans[tc["id"]] = {"command": "", "explanation": f"生成 shell 命令失败：{exc}", "instruction": instruction}
            continue
        plans[tc["id"]] = {
            "command": command,
            "explanation": explanation,
            "instruction": instruction,
        }
    return {"pending_shell": plans}


def _tools_node(state: SearchState) -> dict:
    res = []
    mcp_names = _mcp_tool_names()
    for tool_call in state["messages"][-1].tool_calls:
        name = tool_call["name"]
        args = tool_call.get("args") or {}
        if name == SHELL_TOOL_NAME:
            observation = _run_shell_with_confirmation(tool_call, state)
        elif name in tools_by_name:
            try:
                observation = tools_by_name[name].invoke(args)
            except Exception as exc:
                observation = f"工具 {name} 执行失败：{exc}"
        elif name in mcp_names:
            try:
                observation = _mcp_manager().call_tool(name, args)
            except Exception as exc:
                observation = f"MCP 工具 {name} 调用失败：{exc}"
        else:
            observation = f"未知工具：{name}"
        res.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": res, "pending_shell": {}}


def _route_from_answer(state: SearchState):
    if state["step"] != "tool_called":
        return END
    tcs = getattr(state["messages"][-1], "tool_calls", None) or []
    if any(tc.get("name") == SHELL_TOOL_NAME for tc in tcs):
        return "shell_plan"
    return "tools"


def _run_shell_with_confirmation(tool_call: dict, state: SearchState) -> str:
    plan = (state.get("pending_shell") or {}).get(tool_call["id"]) or {}
    command = plan.get("command") or ""
    explanation = plan.get("explanation") or ""
    if not command:
        return explanation or "未能生成有效的 shell 命令"

    danger = check_dangerous(command)
    if danger:
        return f"已拦截高风险命令（{danger}）：{command}"

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
        return f"已取消执行：{approved_command}"
    danger = check_dangerous(approved_command)
    if danger:
        return f"已拦截高风险命令（{danger}）：{approved_command}"
    return execute_shell_command(approved_command)


def _parse_decision(decision, fallback_command: str) -> tuple[bool, str]:
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


def _truthy(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"y", "yes", "true", "1", "approve"}
