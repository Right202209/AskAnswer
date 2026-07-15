# Important Documentation — D1 · 模型路由 / 上下文预算 / 成本闭环 / 评测基线

> **状态**：代码已落地、**未运行验证**（本批为 code-only session，遵循「不跑项目、
> 不装依赖」约束）。强制验证项已登记为总表 G8 组：
> `docs/important-documentation-verification-matrix.md`（勾选只发生在总表）。
> 本文保留背景、设计不变量、配置矩阵与逻辑自查结论。

## 0. 动机（对齐 Agent 编排 JD 的差距分析）

对照「Agent 编排负责人」JD 逐条盘点后，仓库已有能力与缺口如下：

| JD 主题 | 改造前 | 本批动作 |
|---|---|---|
| 多步 Agent 循环 / 错误恢复 / HITL / MCP | ✅ 已具备 | 不动 |
| **多模型路由与调度**（哪个场景用哪个模型、省钱、保质量） | ❌ 全局单模型 | `routing.py`：角色路由 + 跨 provider 回退链 |
| **上下文窗口物理约束 / 长对话维护** | ❌ 全量历史直发 | `context.py`：token 预算器 + 摘要（brief/llm） |
| **成本意识**（缓存、降级、多厂商定价） | 🟡 仅 4 个 OpenAI 价格 | `pricing.py` 多厂商 + 缓存价；audit 提取 cached_tokens；run 级 token 闸门 |
| **Prompt 缓存策略** | ❌ 动态内容穿插 prompt | `answering.py`：稳定前缀排序 + Anthropic `cache_control` |
| **上线前效果预估 / 上线后数据验证** | ❌ 无 | `evals/`：金标集 + 确定性评测 + 成本预估 + 上线后对账方法 |

## 1. 变更清单（文件 → 内容 → 归属）

| 文件 | 状态 | 内容 |
|---|---|---|
| `askanswer/routing.py` | 新增 | 角色路由（answer/classify/evaluate/summarize）、`RoutedModel` 回退链、backend 缓存、`model_for()` / `describe_routes()` |
| `askanswer/context.py` | 新增 | token 估算（CJK 感知）、块级裁剪（工具配对原子性）、digest off/brief/llm、`BudgetResult.digest_text` |
| `askanswer/answering.py` | 新增 | `_answer_node` 从 `_react_internals.py` 拆出：模型路由 + 预算 + 「稳定前缀 → 动态尾部」prompt + Anthropic `cache_control` |
| `askanswer/_react_internals.py` | 修改 | 只留 tools/confirm 管线（375 → 280 行，回到 300 上限内）；`_emit_tool_telemetry` 移至 answering 并回导 |
| `askanswer/react.py` | 修改 | `_answer_node` 改从 `answering` 导入 |
| `askanswer/load.py` | 修改 | 死导入 `langchain_openai.OpenAI` 删除；`inject_llm_callbacks(label=…)` 公开（用量按真实执行模型归因）；新增 `build_backend` / `raw_backend` |
| `askanswer/nodes.py` | 修改 | 意图分类走 `ROLE_CLASSIFY`；sorcery 前置 run 级 token 成本闸门（`budget_stop` 审计事件） |
| `askanswer/intents/search.py` | 修改 | LLM-as-judge 评估走 `ROLE_EVALUATE` |
| `askanswer/audit.py` | 修改 | `_extract_usage` 三元组（含 cached）：OpenAI `prompt_tokens_details.cached_tokens` / Anthropic `cache_read_input_tokens` / LC `input_token_details.cache_read`；`run_usage_so_far()` |
| `askanswer/pricing.py` | 重写 | `(input, cached_input, output)` 三元价格 × 5 厂商（OpenAI/Anthropic/Google/DeepSeek/Qwen）；`estimate_cost_usd(cached_input_tokens=…)` |
| `askanswer/cli.py` | 修改 | `/status` 新增 `routes` 行（仅非默认路由时展示）；新函数 `_routes_row` |
| `evals/*` | 新增 | 金标集 32 条 + runner + README（指标门槛预注册） |

## 2. 设计不变量（改动时不得破坏）

1. **默认零回归**：不设任何新环境变量时 —— 所有角色解析为全局 `_ModelProxy`（`/model`
   热替换语义不变）、不做历史裁剪、不生成摘要、无成本闸门、`/status` 无 routes 行。
2. **单次注入**：审计/telemetry callback 每条 LLM 调用恰好注入一次。`RoutedModel`
   只用 `raw_backend()`（裸 backend），绝不包裹 `_ModelProxy`（否则双重注入、用量翻倍）。
3. **归因到真实执行者**：回退发生后 token 用量记在实际执行调用的模型标签下
   （`inject_llm_callbacks(label=候选标签)`），不是路由入口的标签。
4. **回退可观测**：每次候选失败写 `kind="model_fallback"` 审计事件（label=失败候选，
   args_summary=`role=…`）。注意：run 上下文之外（无 thread_id）事件会被丢弃 —— 现有
   audit 语义，不在本批扩大。
5. **裁剪不落库**：`budget_messages` 只影响本次发送内容，`state["messages"]` 与
   checkpointer 保持完整历史（时间旅行 / 审计 / 回放不受影响）。
6. **工具配对原子性**：带 `tool_calls` 的 AIMessage 与其后连续 ToolMessage 整块保留或
   整块丢弃；摘要以文本并入 system prompt 动态尾部（mid-list SystemMessage 在
   Anthropic 会报错）。
7. **prompt 缓存友好排序**：system prompt = 稳定前缀（跨请求一致）→ 工具清单（同
   intent 稳定）→ 动态尾部（查询解析 / hint / retry 指令 / 历史摘要）。**改 prompt 时
   新增的动态内容只能加到尾部。**
8. **价格表诚实性**：未登记标签返回 `None`（显示 token 不编造价格）；默认模型
   `openai:gpt-5.4` 刻意未登记（价格未核实），登记前 `/usage` 只展示 token。

## 3. 配置矩阵（全部可选，未设 = 关闭）

| 环境变量 | 作用 | 示例 |
|---|---|---|
| `ASKANSWER_MODEL_CLASSIFY` | 意图分类模型（短输入+结构化，小模型足够） | `openai:gpt-4o-mini` |
| `ASKANSWER_MODEL_EVALUATE` | sorcery 质量评估模型（LLM-as-judge） | `deepseek:deepseek-chat` |
| `ASKANSWER_MODEL_SUMMARIZE` | 历史摘要模型（digest=llm 时消费） | `openai:gpt-4o-mini` |
| `ASKANSWER_MODEL_ANSWER` | 主回答模型（默认=跟随 `/model`） | `anthropic:claude-sonnet-4-5` |
| `ASKANSWER_MODEL_FALLBACKS_<ROLE>` | 逗号分隔回退链（provider 级故障切换） | `openai:gpt-4o,deepseek:deepseek-chat` |
| `ASKANSWER_CONTEXT_MAX_TOKENS` | answer 节点历史预算（正整数；非法/≤0 = 关闭） | `24000` |
| `ASKANSWER_CONTEXT_DIGEST` | 被裁历史摘要：`brief`（确定性）/ `llm`（走 SUMMARIZE 路由，失败回退 brief） | `brief` |
| `ASKANSWER_RUN_TOKEN_BUDGET` | 单轮 run token 上限：超出后 sorcery 跳过质量重试直接收尾 | `60000` |

推荐组合（省钱基线）：`CLASSIFY=EVALUATE=SUMMARIZE=openai:gpt-4o-mini` +
`FALLBACKS_ANSWER=openai:gpt-4o` + `CONTEXT_MAX_TOKENS=24000` + `DIGEST=brief`。
理由：分类/评估/摘要是短输出或判别型任务，mini 档质量损失可忽略而单价差 ~16×；
主回答保旗舰、配同厂回退，先解决可用性再谈跨厂容灾。

## 4. 逻辑自查结论（代码级，未运行）

- 循环导入：`routing → load/audit`，`context →（lazy）routing`，`answering →
  context/routing/load/intents/registry`，`_react_internals → answering`。无环；
  `graph.py --graph` 路径仍不触发 persistence 初始化（audit 顶层只 import 模块，
  `get_persistence()` 调用点均在函数内）。
- 双重注入排查：`RoutedModel` 路径 `raw_backend()/build_backend()` 均为裸 backend ✓；
  默认路径返回 `_ModelProxy` 本体，注入逻辑不变 ✓。
- `stream` 回退语义：仅首分片产出前失败才切换候选（`next(iterator, sentinel)`），
  产出后异常向上抛 —— 避免向用户重复输出前半段。
- 裁剪边界：`kept` 至少含最新块（`if kept and …` 先置 True 才能 break）；
  `dropped = rest[:len(rest)-kept_count]`，块由连续区间构成，切片与块集合一致。
- 兼容性：`estimate_cost_usd` 前三个位置参数不变（cli.py 现有调用不受影响）；
  `log_event` 新参数 `cached_tokens` 有默认值；`persistence.log_audit_events` 按白名单
  键取值，多出的 `cached_tokens` 键被安全忽略（读了 `log_audit_events` 源码确认）。
- 硬性质量指标：新增/修改函数均 ≤50 行、位置参数 ≤3、嵌套 ≤3、常量具名。
  **已知偏离（均为存量、本批未触碰其函数体）**：`cli.py`（2510 行）与
  `persistence.py`（875 行）超 300 行文件上限（cli 拆分已排期为 C1）；
  `_react_internals._tools_node`（113 行）超 50 行函数上限 —— 本批仅从该文件
  移出其他函数（375 → 280 行），`_tools_node` 字节未变，重构它属于 HITL 关键
  路径改造，应在可运行验证的会话里单独做。

## 5. 需要运行验证的事项（登记为总表 G8，勾选在总表）

摘要（完整枚举见总表）：默认零回归冒烟（不设新 env 时 mermaid 拓扑不变、行为不变）、
路由生效与回退链触发（含 `model_fallback` 事件落库）、用量归因标签正确、裁剪后
消息序列合法（无孤儿 ToolMessage）、digest 三模式、成本闸门触发写 `budget_stop`、
`estimate_cost_usd` 缓存折扣计算、评测 runner 全绿且报告数字与金标集一致、
`python -m compileall askanswer` 通过。

## 6. 后续项（本批刻意不做，防止范围膨胀）

1. **persistence schema v5**：`audit_event` 加 `cached_tokens` 列（`_add_column_if_missing`
   一行 + 版本号 +1），让 `/usage` 能展示缓存命中节省。未做原因：persistence.py 875 行
   超文件上限，应随其自身拆分一并处理。
2. **子图角色化**：helix/research/decision/sql 仍走全局模型（answer 形状的工作）。若
   评测显示 `plan_queries`/`interview` 用小模型无损，再引入子图级角色 —— 先测后配。
3. **`/routes` slash 命令**：等 C1 拆完 cli.py 再加（现 `/status` 已可见非默认路由）。
4. **Batch API**：离线评测/回填类任务的半价通道，待出现真实批量场景再接。
5. **Anthropic cache_control 消息级断点**：现只标 system 稳定前缀；历史消息级断点
   需要与裁剪边界联动，等 G8 验证通过后做。
