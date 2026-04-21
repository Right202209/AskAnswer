from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from .load import model, tavily_client
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


_LABEL_MAP = {
    "意图": "intent",
    "文件路径": "file_path",
    "搜索词": "search_query",
    "理解": "understanding",
}


def _mcp_tool_specs() -> list[dict]:
    """Convert registered MCP tools into OpenAI-style function-call dict specs.

    ``bind_tools`` on OpenAI-compatible models accepts these alongside
    LangChain ``BaseTool`` objects, so we can extend the model's toolset
    at runtime without wrapping each MCP tool in pydantic.

     # try use langchain-mcp-adapters to enables agents to use tools defined on MCP servers
    """
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


def _parse_labeled(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        for zh, key in _LABEL_MAP.items():
            matched = False
            for sep in ("：", ":"):
                prefix = zh + sep
                if line.startswith(prefix):
                    out[key] = line[len(prefix):].strip()
                    matched = True
                    break
            if matched:
                break
    return out


def understand_query_node(state: SearchState) -> dict:
    user_message = state["messages"][-1].content
    prompt = (
        f'分析用户的查询："{user_message}"\n\n'
        "请判断用户意图属于以下哪一类：\n"
        "- file_read：要求读取或分析本地文件（通常会给出路径或文件名，"
        "如 /tmp/a.txt、./data.csv、report.md）\n"
        "- chat：闲聊、常识性问题，或你已知可直接回答、不需要联网\n"
        "- search：需要联网搜索获取实时、最新或不确定的信息\n\n"
        "严格按以下格式输出，没有内容的字段留空：\n"
        "意图：file_read|chat|search\n"
        "文件路径：（仅 file_read 时填写具体路径）\n"
        "搜索词：（仅 search 时填写最佳关键词）\n"
        "理解：（对用户需求的简要总结）\n"
    )
    response = model.invoke([SystemMessage(content=prompt)])
    fields = _parse_labeled(response.content)

    intent = (fields.get("intent") or "").lower().strip()
    if intent not in {"file_read", "search", "chat"}:
        intent = "search"
    search_query = fields.get("search_query") or user_message
    file_path = fields.get("file_path") or ""
    understanding = fields.get("understanding") or user_message

    if intent == "file_read":
        hint = f"识别为读文件：{file_path or '(未能提取路径)'}"
    elif intent == "chat":
        hint = "识别为闲聊/常识问题，直接回答"
    else:
        hint = f"识别为搜索，关键词：{search_query}"

    return {
        "user_query": understanding,
        "search_query": search_query,
        "file_path": file_path,
        "intent": intent,
        "retry_count": 0,
        "step": "understood",
        "messages": [AIMessage(content=hint)],
    }


def file_read_node(state: SearchState) -> dict:
    path = (state.get("file_path") or "").strip()
    if not path:
        msg = "未能识别要读取的文件路径，请明确指出文件路径后重试。"
        return {
            "final_answer": msg,
            "step": "completed",
            "messages": [AIMessage(content=msg)],
        }
    try:
        content = tools_by_name["read_file"].invoke({"path": path})
    except Exception as exc:
        content = f"读取 `{path}` 失败：{exc}"
    return {
        "final_answer": str(content),
        "step": "completed",
        "messages": [AIMessage(content=str(content))],
    }


def tavily_search_node(state: SearchState) -> dict:
    search_query = state["search_query"]
    try:
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )

        results = response.get("results", [])
        tavily_answer = (response.get("answer") or "").strip()

        search_results = f"查询关键词：{search_query}\n\n"
        if tavily_answer:
            search_results += f"Tavily 摘要：{tavily_answer}\n\n"
        search_results += "搜索结果（Top 5）：\n\n"

        if results:
            for i, result in enumerate(results, 1):
                title = result.get("title", "无标题")
                url = result.get("url", "#")
                content = result.get("content", "无内容摘要")
                score = result.get("score", 0.0)

                search_results += (
                    f"{i}. **{title}** (相关度: {score:.3f})\n"
                    f"   链接: {url}\n"
                    f"   {content[:280]}{'...' if len(content) > 280 else ''}\n\n"
                )
        else:
            search_results += "未找到任何搜索结果。\n"

        return {
            "search_results": search_results.strip(),
            "step": "searched",
            "messages": [AIMessage(content="搜索完成~ 正在整理答案...")],
        }
    except Exception as e:
        return {
            "search_results": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": [AIMessage(content=" 搜索遇到问题...")],
        }


def generate_answer_node(state: SearchState) -> dict:
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


def sorcery_answer_node(state: SearchState) -> dict:
    intent = state.get("intent", "search")
    final_answer = state.get("final_answer", "")

    # 只有 search 路径允许改写搜索词重跑；file_read / chat 直接通过
    if intent != "search":
        return {
            "final_answer": final_answer,
            "step": "completed",
            "messages": [AIMessage(content=final_answer)],
        }

    retry_count = state.get("retry_count", 0)
    if retry_count >= 1:
        return {
            "final_answer": final_answer,
            "step": "completed",
            "messages": [AIMessage(content=final_answer)],
        }

    prompt = f"""请评估当前答案是否足够好。
    用户问题：{state.get('user_query', '')}
    当前搜索词：{state.get('search_query', '')}
    当前搜索结果：{state.get('search_results', '')}
    当前答案：{final_answer}

    如果答案已经足够好，请严格输出：
    评分：pass
    新搜索词：

    如果答案不够好，需要重新搜索，请严格输出：
    评分：re_search
    新搜索词：[更精准的新搜索词]
    """
    response = model.invoke([SystemMessage(content=prompt)])
    response_text = response.content

    if "评分：re_search" in response_text and "新搜索词：" in response_text:
        new_search_query = response_text.split("新搜索词：", 1)[1].strip()
        if new_search_query:
            return {
                "search_query": new_search_query,
                "retry_count": retry_count + 1,
                "step": "retry_search",
                "messages": [
                    AIMessage(content=f"当前答案不够理想，改为搜索：{new_search_query}")
                ],
            }

    return {
        "final_answer": final_answer,
        "step": "completed",
        "messages": [AIMessage(content=final_answer)],
    }


def shell_plan_node(state: SearchState) -> dict:
    """为本轮 tool_calls 中的 shell 调用预生成命令，写入 state。

    该节点返回后 checkpointer 会持久化 ``pending_shell``，随后 ``tools_node``
    即使因 ``interrupt()`` 被重放，也只读取 state，不再重新调用 LLM。
    """
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


def tools_node(state: SearchState) -> dict:
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

