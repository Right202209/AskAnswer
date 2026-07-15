"""Model pricing table for local usage summaries (multi-provider).

Values are USD per 1K tokens as ``(input, cached_input, output)``.
``cached_input=None`` 表示该模型无缓存折扣价（缓存部分按 input 价计）。
Unknown labels intentionally return ``None`` so the CLI can still show token
totals without pretending to know a price.

维护约定：
- key 与 audit 的 ``model_label`` 一致（``provider:model``，即 ``init_chat_model``
  的 provider 前缀）。通过 OpenAI 兼容端点接入的第三方模型（如 Qwen）会带
  ``openai:`` 前缀，按实际标签登记。
- 价格是 2026-07 的**指示性快照**，用于本地成本归因与路由决策对比；接入真实
  计费/预算控制前必须对照厂商价目页复核（见 important-documentation-d1）。
"""

from __future__ import annotations


_TOKENS_PER_UNIT = 1000

_USD_PER_1K: dict[str, tuple[float, float | None, float]] = {
    # OpenAI
    "openai:gpt-4o": (0.0025, 0.00125, 0.0100),
    "openai:gpt-4o-mini": (0.00015, 0.000075, 0.00060),
    "openai:gpt-4.1": (0.0020, 0.00050, 0.0080),
    "openai:gpt-4.1-mini": (0.00040, 0.00010, 0.00160),
    # Anthropic（cached_input 为 cache-read 价；cache 写入的 1.25x 溢价不在此表建模）
    "anthropic:claude-sonnet-4-5": (0.0030, 0.00030, 0.0150),
    "anthropic:claude-haiku-4-5": (0.0010, 0.00010, 0.0050),
    "anthropic:claude-opus-4-1": (0.0150, 0.00150, 0.0750),
    # Google
    "google_genai:gemini-2.5-flash": (0.00030, 0.000075, 0.00250),
    "google_genai:gemini-2.5-pro": (0.00125, 0.00031, 0.0100),
    # DeepSeek（cached_input 为 cache-hit 价）
    "deepseek:deepseek-chat": (0.00027, 0.00007, 0.00110),
    "deepseek:deepseek-reasoner": (0.00055, 0.00014, 0.00219),
    # Qwen（DashScope 的 OpenAI 兼容端点，label 带 openai: 前缀）
    "openai:qwen-plus": (0.00040, None, 0.00120),
    "openai:qwen-max": (0.00160, None, 0.00640),
}


def estimate_cost_usd(
    model_label: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    cached_input_tokens: int | None = None,
) -> float | None:
    """按标签估算一次/一批调用的成本；未知标签返回 None（不编造价格）。

    ``cached_input_tokens`` 是 input 中命中 prompt cache 的部分（audit 的
    ``cached_tokens`` 字段），按缓存折扣价计；超出 input 总量时按 input 截断。
    """
    if not model_label:
        return None
    prices = _USD_PER_1K.get(model_label)
    if prices is None:
        return None
    input_price, cached_price, output_price = prices
    input_total = input_tokens or 0
    cached = min(cached_input_tokens or 0, input_total)
    fresh = input_total - cached
    effective_cached_price = cached_price if cached_price is not None else input_price
    return (
        fresh / _TOKENS_PER_UNIT * input_price
        + cached / _TOKENS_PER_UNIT * effective_cached_price
        + (output_tokens or 0) / _TOKENS_PER_UNIT * output_price
    )


def known_labels() -> tuple[str, ...]:
    """已登记价格的模型标签（评测报告用来标注「可估价/不可估价」）。"""
    return tuple(_USD_PER_1K)


def format_cost(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "$0"
    return f"${value:.4f}"
