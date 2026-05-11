"""Tiny model pricing table for local usage summaries.

Values are USD per 1K tokens.  Unknown models intentionally return ``None`` so
the CLI can still show token totals without pretending to know a price.
"""

from __future__ import annotations


_USD_PER_1K: dict[str, tuple[float, float]] = {
    "openai:gpt-4o": (0.0025, 0.0100),
    "openai:gpt-4o-mini": (0.00015, 0.00060),
    "openai:gpt-4.1": (0.0020, 0.0080),
    "openai:gpt-4.1-mini": (0.00040, 0.00160),
}


def estimate_cost_usd(
    model_label: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if not model_label:
        return None
    prices = _USD_PER_1K.get(model_label)
    if prices is None:
        return None
    in_price, out_price = prices
    return ((input_tokens or 0) / 1000 * in_price) + ((output_tokens or 0) / 1000 * out_price)


def format_cost(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "$0"
    return f"${value:.4f}"
