"""react 子图的 answer 节点：模型路由、上下文预算、缓存友好的 prompt 组装。

从 ``_react_internals.py`` 拆出的「回答策略」层 —— 该文件只保留 tools /
confirm 执行管线。这里集中三件与 LLM 调用形状相关的事：

1. **模型路由**：主推理走 ``model_for(ROLE_ANSWER)``（默认=全局 `/model`，
   可用环境变量配置差异化模型与回退链，见 ``routing.py``）。
2. **上下文预算**：发送前用 ``context.budget_messages`` 把历史裁进
   ``ASKANSWER_CONTEXT_MAX_TOKENS``（未设置时零开销直通）。
3. **prompt 前缀稳定性**：system prompt 按「跨请求稳定 → 按 intent 半稳定 →
   逐请求动态」排序。OpenAI 的隐式 prompt cache 与 Anthropic 的显式
   ``cache_control`` 都按**前缀**命中 —— 把易变内容放尾部是缓存省钱的前提。
   provider 为 anthropic 时，稳定段落会带上 ``cache_control`` 断点标记。
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .context import answer_token_budget, budget_messages, digest_mode
from .intents import get_intent_registry
from .load import current_model_label
from .registry import get_registry
from .routing import ROLE_ANSWER, model_for
from .state import SearchState

_ANTHROPIC_PROVIDER = "anthropic"

# 跨请求完全一致的开场白：放在 prompt 最前，最大化 provider 前缀缓存命中。
_STABLE_PREAMBLE = (
    "你是 AskAnswer 助手，可以调用工具来协助用户。\n"
    "若用户需要相应信息，直接调用对应工具；否则结合上下文直接回答。\n"
    "回答使用用户的语言，事实性内容注明依据（工具结果或已有知识）。"
)


def _emit_tool_telemetry(
    *,
    tool_name: str,
    duration_ms: int | None = None,
    error: str | None = None,
) -> None:
    """把一次工具调用发到 telemetry（独立于 audit 写入路径，共享最小 schema）。

    lazy import：本模块顶层不依赖 telemetry（对齐 execution-plan 2.2 的不变量）；
    telemetry 未启用时 ``emit_event`` 内部会立即返回，零开销。
    """
    from . import telemetry

    telemetry.emit_event(
        kind="tool_call",
        tool_name=tool_name,
        duration_ms=duration_ms,
        error=error,
    )


def _reclassify_intent(state: SearchState) -> str | None:
    """根据「最新一条用户消息」重新判定 intent，让会话中途切换主题时也能换工具集。

    返回新的 intent 字符串；返回 ``None`` 表示这次不需要切换（例如最新一条不是
    新的真人输入，或本地分类器拿不准）。
    """
    if state.get("step") == "retry_search":
        # sorcery 触发的“重新搜索”重试不是新一轮提问，原始 intent 已在
        # understand 阶段确定，跳过避免反转。
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


def _stable_block(tool_names: str) -> str:
    """稳定 + 半稳定段：开场白（全局一致）+ 工具清单（同 intent 内一致）。"""
    return f"{_STABLE_PREAMBLE}\n当前可用工具：{tool_names}。"


def _dynamic_block(*, state: SearchState, context_line: str, digest_text: str) -> str:
    """逐请求变化的段落，永远放在 system prompt 尾部。

    被裁掉的历史摘要（若有）并入这里，而非作为独立消息插回列表 —— 保证
    system prompt 的稳定前缀不被打断，也避开 mid-list SystemMessage 的兼容问题。
    """
    parts = [f"用户查询解析：{state.get('user_query', '')}", context_line]
    if digest_text:
        parts.append(digest_text)
    return "\n".join(p for p in parts if p)


def _build_system_message(
    *,
    provider: str,
    stable: str,
    dynamic: str,
) -> SystemMessage:
    """anthropic → 分块 content + ``cache_control`` 断点（稳定前缀显式可缓存）；
    其余 provider → 单字符串（稳定前缀在前，隐式前缀缓存同样受益）。"""
    if provider == _ANTHROPIC_PROVIDER:
        return SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": stable,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": dynamic},
            ]
        )
    return SystemMessage(content=f"{stable}\n\n{dynamic}")


def _provider_of(llm) -> str:
    """从模型句柄取 provider 名：RoutedModel 有 label，代理回退到全局标签。"""
    label = getattr(llm, "label", None) or current_model_label()
    return str(label).split(":", 1)[0]


def _answer_output(
    *,
    state: SearchState,
    response,
    new_intent: str | None,
    had_retry_directive: bool,
) -> dict:
    """把 LLM 响应整理成节点返回的部分状态（分「继续调工具」与「产出答案」两路）。"""
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        out: dict = {"step": "tool_called", "messages": [response]}
    else:
        out = {
            "final_answer": response.content,
            "step": "completed",
            "messages": [response],
        }
    if had_retry_directive:
        out["retry_directive"] = {}
    # 中途切换 intent 时，把新 intent 写回 state，下一轮 tool 选择就会变化
    if new_intent and new_intent != state.get("intent"):
        out["intent"] = new_intent
    return out


def _answer_node(state: SearchState) -> dict:
    """react 主推理节点：路由模型、按预算裁历史、拼 system prompt、调用 LLM。"""
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

    # 历史裁剪只影响本次发送的消息，state["messages"] 原样保留（审计/回放不受影响）
    budgeted = budget_messages(
        list(state["messages"]),
        max_tokens=answer_token_budget(),
        digest=digest_mode(),
    )
    llm = model_for(ROLE_ANSWER)
    system_msg = _build_system_message(
        provider=_provider_of(llm),
        stable=_stable_block(tool_names),
        dynamic=_dynamic_block(
            state=state,
            context_line=context_line,
            digest_text=budgeted.digest_text,
        ),
    )

    bound = llm.bind_tools(bundle_tools) if bundle_tools else llm
    response = bound.invoke([system_msg] + budgeted.messages)
    return _answer_output(
        state=state,
        response=response,
        new_intent=new_intent,
        had_retry_directive=bool(retry_directive),
    )
