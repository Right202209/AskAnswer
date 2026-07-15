"""上下文窗口预算：把发给 LLM 的消息历史裁到 token 预算内。

为什么需要它：``_answer_node`` 会把 ``[System] + 全部历史`` 一次性发给模型。
多轮对话或长工具输出会让历史无界增长，最终撞上上下文窗口上限（报错或悄悄
截断）。从「Agent 视角」看，这是必须显式管理的物理约束，而不是等 provider
报错。本模块提供一个确定性、默认零 LLM 调用、无新依赖的预算器；被裁掉的
历史可按需生成摘要（brief=确定性拼接 / llm=走 ROLE_SUMMARIZE 路由）。

三条不变量（对齐 docs/important-documentation-d1-routing-context-cost-eval.md）：
1. 默认零回归：``max_tokens=None`` 时原样返回，且不做任何 token 估算（零开销）。
   通过 ``ASKANSWER_CONTEXT_MAX_TOKENS`` 显式开启。
2. 保留 system 前缀与最新一条：分类/意图提示在 system 里，最后一条是当前轮，
   两者永不裁剪。
3. 工具调用配对完整：带 ``tool_calls`` 的 AIMessage 与其后的 ToolMessage 作为
   一个原子块整体保留或整体丢弃 —— 绝不产生「孤儿 ToolMessage」（多数 provider
   会因此报 400）。摘要以**文本**形式返回而非插回消息列表：mid-list
   SystemMessage 在部分 provider（如 Anthropic）会直接报错。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, SystemMessage

# 每条消息除正文外的固定开销（role 标记、分隔符等），参考 OpenAI 计费口径。
_PER_MESSAGE_OVERHEAD = 4
# 字符→token 的粗略换算：非 CJK 约 4 字符/token，CJK 约 1.5 字符/token。
# 估算刻意偏保守（略高估），宁可多裁一点也不要溢出窗口。
_NON_CJK_CHARS_PER_TOKEN = 4.0
_CJK_CHARS_PER_TOKEN = 1.5
_CJK_RANGES = (
    (0x4E00, 0x9FFF),   # CJK 统一表意
    (0x3040, 0x30FF),   # 平假名 + 片假名
    (0xAC00, 0xD7AF),   # 谚文音节
    (0x3400, 0x4DBF),   # CJK 扩展 A
)
_DEFAULT_MAX_TOKENS_ENV = "ASKANSWER_CONTEXT_MAX_TOKENS"
_SUMMARIZE_ENV = "ASKANSWER_CONTEXT_DIGEST"
_DIGEST_ITEM_CHARS = 80
_DIGEST_MAX_ITEMS = 12
_LLM_DIGEST_MAX_ITEMS = 40
_LLM_DIGEST_MAX_CHARS = 1200

# 摘要模式：off = 只裁不摘；brief = 确定性摘要（零 LLM 成本）；
# llm = 用 ROLE_SUMMARIZE 路由生成要点（失败自动回退 brief）。
DIGEST_OFF = "off"
DIGEST_BRIEF = "brief"
DIGEST_LLM = "llm"


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """字符级 token 估算（无 tokenizer 依赖，保守偏高）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    est = cjk / _CJK_CHARS_PER_TOKEN + other / _NON_CJK_CHARS_PER_TOKEN
    return int(est) + 1


def _content_text(message: BaseMessage) -> str:
    """把消息正文规约成字符串（content 可能是 str 或 分块 list）。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
        return " ".join(parts)
    return str(content)


def message_tokens(message: BaseMessage) -> int:
    """单条消息的 token 估算：正文 + 工具调用序列化 + 固定开销。"""
    total = estimate_tokens(_content_text(message)) + _PER_MESSAGE_OVERHEAD
    for call in getattr(message, "tool_calls", None) or ():
        name = call.get("name", "") if isinstance(call, dict) else ""
        raw_args = call.get("args", "") if isinstance(call, dict) else ""
        total += estimate_tokens(f"{name}{raw_args}")
    return total


@dataclass(frozen=True)
class BudgetResult:
    messages: list[BaseMessage]
    dropped: int = 0
    kept_tokens: int = 0
    # 被丢弃历史的摘要文本（digest != off 且确有丢弃时非空）。
    # 刻意不做成消息插回列表：mid-list SystemMessage 在部分 provider（如
    # Anthropic）会报错，由调用方并入首条 system prompt 的动态尾部。
    digest_text: str = ""
    dropped_labels: list[str] = field(default_factory=list)


def _is_tool_message(message: BaseMessage) -> bool:
    return getattr(message, "type", None) == "tool"


def _has_tool_calls(message: BaseMessage) -> bool:
    return bool(getattr(message, "tool_calls", None))


def _group_blocks(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """把消息切成原子块：带 tool_calls 的 AIMessage 与其后的 ToolMessage 同组。

    这样裁剪只会在块边界发生，永远不会把 tool_call 与它的 ToolMessage 拆开。
    """
    blocks: list[list[BaseMessage]] = []
    i = 0
    n = len(messages)
    while i < n:
        current = messages[i]
        if _has_tool_calls(current):
            block = [current]
            i += 1
            while i < n and _is_tool_message(messages[i]):
                block.append(messages[i])
                i += 1
            blocks.append(block)
        else:
            blocks.append([current])
            i += 1
    return blocks


def _split_system_prefix(
    messages: list[BaseMessage],
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    idx = 0
    for message in messages:
        if isinstance(message, SystemMessage):
            idx += 1
        else:
            break
    return messages[:idx], messages[idx:]


def _label(message: BaseMessage) -> str:
    kind = getattr(message, "type", "msg")
    text = _content_text(message).strip().replace("\n", " ")
    if len(text) > _DIGEST_ITEM_CHARS:
        text = text[: _DIGEST_ITEM_CHARS - 1] + "…"
    return f"{kind}: {text}" if text else kind


def _brief_digest(dropped: list[BaseMessage]) -> str:
    """确定性摘要：不调用 LLM，仅取角色 + 截断正文，零成本零不确定性。"""
    lines = [_label(m) for m in dropped[-_DIGEST_MAX_ITEMS:]]
    body = "\n".join(f"- {line}" for line in lines)
    return f"[已省略 {len(dropped)} 条更早的历史，要点如下]\n{body}"


def _llm_digest(dropped: list[BaseMessage]) -> str | None:
    """用 ROLE_SUMMARIZE 路由把被丢弃历史压成要点；任何失败返回 None（调用方回退 brief）。

    lazy import routing：避免 context.py 顶层引入模型层（保持可单测、无副作用）。
    摘要本身也走角色路由 —— 长输入 + 短输出，是「小模型跑便宜活」的典型场景。
    """
    from langchain_core.messages import SystemMessage as _Sys

    from .routing import ROLE_SUMMARIZE, model_for

    transcript_parts = []
    for message in dropped[-_LLM_DIGEST_MAX_ITEMS:]:
        role = getattr(message, "type", "msg")
        transcript_parts.append(f"[{role}] {_content_text(message)}")
    transcript = "\n".join(transcript_parts)[:_LLM_DIGEST_MAX_CHARS * 6]
    prompt = (
        "把以下对话历史压缩成不超过 8 条要点，保留人物、决定、约束、已知事实与"
        "待办；丢弃寒暄与冗余。用中文，直接输出要点列表。\n\n" + transcript
    )
    try:
        response = model_for(ROLE_SUMMARIZE).invoke([_Sys(content=prompt)])
    except Exception:
        return None
    text = str(getattr(response, "content", "") or "").strip()
    if not text:
        return None
    return f"[已省略 {len(dropped)} 条更早的历史，摘要如下]\n{text[:_LLM_DIGEST_MAX_CHARS]}"


def _make_digest(dropped: list[BaseMessage], mode: str) -> str:
    if not dropped or mode == DIGEST_OFF:
        return ""
    if mode == DIGEST_LLM:
        return _llm_digest(dropped) or _brief_digest(dropped)
    return _brief_digest(dropped)


def budget_messages(
    messages: list[BaseMessage],
    *,
    max_tokens: int | None = None,
    digest: str = DIGEST_OFF,
) -> BudgetResult:
    """把消息裁到 ``max_tokens`` 预算内，保留 system 前缀、最新一条与工具配对。

    ``max_tokens=None`` → 原样返回且不做估算（零开销快路径）。
    ``digest`` ∈ {off, brief, llm}：对被丢弃的历史生成摘要文本（放在
    ``BudgetResult.digest_text``，由调用方并入 system prompt，不插回消息列表）。
    """
    if max_tokens is None or not messages:
        return BudgetResult(messages=messages)

    system, rest = _split_system_prefix(messages)
    system_tokens = sum(message_tokens(m) for m in system)
    remaining = max_tokens - system_tokens
    blocks = _group_blocks(rest)

    kept: list[list[BaseMessage]] = []
    used = 0
    # 从最新块往回收，保证最近上下文优先保留；最新块即便超预算也保留（当前轮）。
    for block in reversed(blocks):
        cost = sum(message_tokens(m) for m in block)
        if kept and used + cost > remaining:
            break
        kept.append(block)
        used += cost
    kept.reverse()

    kept_flat = [m for block in kept for m in block]
    kept_count = len(kept_flat)
    dropped_msgs = rest[: len(rest) - kept_count] if kept_count else rest
    return BudgetResult(
        messages=list(system) + kept_flat,
        dropped=len(dropped_msgs),
        kept_tokens=system_tokens + used,
        digest_text=_make_digest(dropped_msgs, digest),
        dropped_labels=[_label(m) for m in dropped_msgs],
    )


def answer_token_budget() -> int | None:
    """从环境读取 answer 节点的上下文预算；未设置返回 None（不启用裁剪）。"""
    raw = os.environ.get(_DEFAULT_MAX_TOKENS_ENV, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def digest_mode() -> str:
    """摘要模式：off（默认）/ brief / llm。未识别值一律回退 off（非回归）。"""
    raw = os.environ.get(_SUMMARIZE_ENV, "").strip().lower()
    if raw in ("1", "true", "yes", "on", DIGEST_BRIEF):
        return DIGEST_BRIEF
    if raw == DIGEST_LLM:
        return DIGEST_LLM
    return DIGEST_OFF
