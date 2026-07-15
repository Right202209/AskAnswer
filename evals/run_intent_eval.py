#!/usr/bin/env python3
"""意图分类金标集评测：上线前效果预估的确定性部分（免 API key、可重复）。

跑什么：对 ``golden_intents.jsonl`` 的每条用例执行 ``classify_local``（与
``understand_query_node`` 首选路径完全相同的代码），产出：

- 意图准确率（known_gap 用例单独统计，不计入准确率）；
- 本地覆盖率 = 不需要 LLM 兜底的比例（直接决定分类环节的成本与延迟）；
- 本地分类延迟 mean / p50 / p95；
- 分类环节成本预估：classify 路由模型 vs answer 旗舰模型的每千次对比。

不跑什么：LLM 兜底分类与端到端回答质量 —— 这部分要花钱且不确定，属于
evals/README.md 里的「上线后用 /usage 与 audit 数据验证」闭环。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_intents.jsonl"

# LLM 兜底分类一次调用的 token 画像（prompt 模板 + 常见问句长度的经验值，
# 用于成本预估；与真实值的偏差会同比例作用于两边，不影响“对比结论”）。
CLASSIFY_INPUT_TOKENS_EST = 420
CLASSIFY_OUTPUT_TOKENS_EST = 60
PROJECTION_QUERIES = 1000
P95 = 0.95


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def run_case(registry, case: dict) -> dict:
    query = case["query"]
    started = time.perf_counter()
    fields = registry.classify_local(query)
    latency_ms = (time.perf_counter() - started) * 1000
    predicted = fields.intent if fields is not None else None
    expects_llm = bool(case.get("expects_llm"))
    expected = case.get("expected")
    ok = (predicted is None) if expects_llm else (predicted == expected)
    return {
        "id": case.get("id", "?"),
        "query": query,
        "expected": "（LLM 兜底）" if expects_llm else expected,
        "predicted": predicted or "（LLM 兜底）",
        "ok": ok,
        "known_gap": bool(case.get("known_gap")),
        "needs_llm": predicted is None,
        "latency_ms": latency_ms,
        "note": case.get("note", ""),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * q), len(ordered) - 1)
    return ordered[index]


def aggregate(rows: list[dict]) -> dict:
    scored = [r for r in rows if not r["known_gap"]]
    gaps = [r for r in rows if r["known_gap"]]
    correct = sum(1 for r in scored if r["ok"])
    llm_needed = sum(1 for r in rows if r["needs_llm"])
    latencies = [r["latency_ms"] for r in rows]
    return {
        "total": len(rows),
        "scored": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else 0.0,
        "llm_rate": llm_needed / len(rows) if rows else 0.0,
        "local_coverage": 1 - (llm_needed / len(rows)) if rows else 0.0,
        "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, P95),
        "known_gaps_total": len(gaps),
        "known_gaps_reproduced": sum(1 for r in gaps if not r["ok"]),
        "failures": [r for r in scored if not r["ok"]],
        "gap_rows": gaps,
    }


def _role_primary_label(role: str) -> str:
    from askanswer.load import current_model_label
    from askanswer.routing import get_router

    route = get_router().route(role)
    return route.spec or current_model_label()


def cost_projection(llm_rate: float) -> list[str]:
    """按当前路由表预估「每 1000 次查询」的 LLM 兜底分类成本对比。"""
    from askanswer.pricing import estimate_cost_usd, format_cost
    from askanswer.routing import ROLE_ANSWER, ROLE_CLASSIFY

    calls = llm_rate * PROJECTION_QUERIES
    lines = [f"- LLM 兜底率 {llm_rate:.0%} → 每 {PROJECTION_QUERIES} 次查询约 {calls:.0f} 次兜底分类调用"]
    costs: dict[str, float | None] = {}
    for role in (ROLE_CLASSIFY, ROLE_ANSWER):
        label = _role_primary_label(role)
        per_call = estimate_cost_usd(
            label, CLASSIFY_INPUT_TOKENS_EST, CLASSIFY_OUTPUT_TOKENS_EST
        )
        costs[role] = None if per_call is None else per_call * calls
        rendered = "未登记价格（pricing.py）" if per_call is None else format_cost(costs[role])
        lines.append(f"- {role} 路由（{label}）：{rendered}")
    classify_cost, answer_cost = costs[ROLE_CLASSIFY], costs[ROLE_ANSWER]
    if classify_cost is not None and answer_cost:
        saved = 1 - classify_cost / answer_cost
        lines.append(f"- 分类走 classify 路由相对全走 answer 模型：节省 {saved:.0%}")
    return lines


def render_markdown(agg: dict, routes: dict[str, str]) -> str:
    out = ["# 意图分类金标集评测报告", ""]
    out.append(f"- 用例：{agg['total']}（计分 {agg['scored']} + 已知缺口 {agg['known_gaps_total']}）")
    out.append(f"- 准确率：{agg['correct']}/{agg['scored']} = {agg['accuracy']:.1%}")
    out.append(f"- 本地覆盖率：{agg['local_coverage']:.1%}（LLM 兜底率 {agg['llm_rate']:.1%}）")
    out.append(
        f"- 本地分类延迟：mean {agg['latency_mean_ms']:.2f}ms · "
        f"p50 {agg['latency_p50_ms']:.2f}ms · p95 {agg['latency_p95_ms']:.2f}ms"
    )
    gaps = f"{agg['known_gaps_reproduced']}/{agg['known_gaps_total']}"
    out.append(f"- 已知缺口复现：{gaps}（若小于全数=有缺口已修复，请更新金标集）")
    out += ["", "## 成本预估（分类环节）", *cost_projection(agg["llm_rate"])]
    out += ["", "## 当前模型路由", *[f"- {role}: {chain}" for role, chain in routes.items()]]
    if agg["failures"]:
        out += ["", "## 失败明细", "| id | 期望 | 实际 | note |", "|---|---|---|---|"]
        for r in agg["failures"]:
            out.append(f"| {r['id']} | {r['expected']} | {r['predicted']} | {r['note']} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON 而非 Markdown")
    args = parser.parse_args(argv)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    # 放在 main 里 import：加载 askanswer 会构建模型客户端，需要 .env 就绪
    from askanswer.intents import get_intent_registry
    from askanswer.routing import describe_routes

    rows = [run_case(get_intent_registry(), case) for case in load_cases(args.golden)]
    agg = aggregate(rows)
    if args.json:
        payload = {k: v for k, v in agg.items() if k not in ("failures", "gap_rows")}
        payload["rows"] = rows
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(agg, describe_routes()))
    return 0 if not agg["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
