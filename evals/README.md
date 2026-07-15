# Evals — 上线前效果预估与上线后验证

> 角色：把「效果」变成可预估、可复测的数字。上线前跑离线金标集给出预期指标；
> 上线后用 `/usage`、`/audit`（SQLite `audit_event` 表）的真实数据回来对账。

## 目录

- `golden_intents.jsonl` — 意图分类金标集（32 条：8 个 intent × 中英文 + 关键词碰撞用例 + LLM 兜底用例 + 已知缺口用例）
- `run_intent_eval.py` — 评测 runner（离线、确定性、免 API 调用）

## 怎么跑

```bash
source .venv/bin/activate
python evals/run_intent_eval.py            # Markdown 报告到 stdout
python evals/run_intent_eval.py --json     # 机器可读（CI 存档用）
```

注意：import `askanswer` 会在模块加载时构建模型客户端，因此需要 `.env` 里有
`OPENAI_API_KEY` / `TAVILY_API_KEY`（**评测本身不发起任何 API 调用**，key 只需
存在、不校验有效性）。退出码：有计分用例失败 → 1（可直接接 CI）。

## 指标定义与上线门槛（预注册，跑完后填实测值）

| 指标 | 定义 | 门槛 |
|---|---|---|
| 意图准确率 | 计分用例中 `classify_local` 结果 == 期望 intent 的比例（known_gap 不计分） | ≥ 90% |
| 本地覆盖率 | 不需要 LLM 兜底分类的比例。直接决定分类环节的 P50 延迟与成本 | ≥ 60% |
| 本地分类延迟 p95 | `classify_local` 单次耗时（纯正则/关键词，无 IO） | < 5 ms |
| 已知缺口复现 | `known_gap` 用例仍失败的数量。少于全数 = 有缺口被修复，需更新金标集 | = 全数 |

## 成本预估方法（报告的「成本预估」小节）

- 每次 LLM 兜底分类按固定 token 画像估算（输入 420 / 输出 60，`run_intent_eval.py`
  顶部常量）；画像偏差对「classify 路由 vs answer 旗舰」两边同比作用，不影响对比结论。
- 单价来自 `askanswer/pricing.py`（指示性快照）；未登记标签如实显示「未登记价格」。
- 预估公式：`LLM 兜底率 × 1000 次查询 × 单次成本`，输出两条路由的对比与节省比例。

## 上线后验证闭环

1. `/usage --days 7`：按模型标签核对真实 token 用量与成本归因（含 `model_fallback`
   事件频率——回退频繁说明主模型不稳或路由配置不当）。
2. `sqlite3 ~/.askanswer/state.db "SELECT intent, COUNT(*) FROM audit_event WHERE kind='llm_call' GROUP BY intent"`
   —— 真实流量的意图分布，回填金标集配比（金标集应向真实分布收敛）。
3. `kind='budget_stop'` 事件数：成本闸门触发频率，过高说明 `ASKANSWER_RUN_TOKEN_BUDGET`
   设得太紧或某意图重试策略过于激进。

## 维护规则

- 新增 intent → 至少补 3 条用例（中文触发、英文触发、与相邻优先级 handler 的碰撞用例）。
- 发现线上误分类 → 先补一条 `known_gap: true` 用例复现，修分类器后把该条转正
  （删掉 `known_gap` 字段并改期望值）。金标集只增不删，误例即回归测试。
- 期望值必须从 `askanswer/intents/*.py` 的实际触发规则推导（优先级：file_read 10 ·
  sql 20 · decision 21 · helix 22 · math 25 · research 28 · search 30 · chat 40），
  不要凭感觉写。
