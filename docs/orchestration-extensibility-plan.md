# 优化 AskAnswer 编排：以可扩展性为核心的重构方案

## Context

当前主图 `START → understand → answer → sorcery → (END | answer)` 在功能上跑得通，但**新增一个 intent（或一类工具）需要修改 6+ 个文件**，且每个文件里都是 if/elif 分支。这违背 OCP（对扩展开放、对修改封闭），也是 TODO.md 里多条编排相关条目的根本原因。

具体痛点（按"加一个新 intent，比如 `code_analysis`，要改哪些地方"为基线统计）：

| 痛点 | 当前位置 | 影响 |
|---|---|---|
| Intent 名/关键词表/路径正则 | `nodes.py:14-122` | 每加一个 intent，要新增一组常量 + 修改 `_local_intent` 级联 |
| LLM JSON 提示词 | `nodes.py:275-294` | 手写 JSON 解析（`_parse_json_object`），脆弱且要改 prompt |
| `_answer_node` 系统提示 | `_react_internals.py:62-77` | if/elif 决定 `context_line`，新 intent = 新分支 |
| Sorcery 仅对 search 生效 | `nodes.py:342` | `if intent != "search": return` 把 evaluate 锁死给 search |
| 重试用合成 HumanMessage | `nodes.py:381-391` | 字符串模板，无法复用给 file_read/sql 失败重试 |
| 工具 bundle 硬编码 | `registry.py:26-38` | `BUNDLE_CHAT/SEARCH/FILE/SQL` 是常量，不能加正交 tag |
| 确认机制只针对 shell | `registry.py:50` `requires_confirmation: bool` | 想加 fs_write 确认就要改 HITL 整条链 |
| CLI 渲染 per-node if/elif | `cli.py:322-371` | 加 intent / 改进度文案 = 改 CLI |
| `state.intent` / `state.step` 是 str | `state.py:11,17` | 拼写错误零提示，路由分支无 IDE 帮助 |
| Mid-conversation 重分类硬编码 | `_react_internals.py:32-52` | 直接调用 `_local_intent`，没法替换分类策略 |

**目标产出**：把"加 intent / 加工具 / 加 HITL 类别"从"改 6 个文件 + 谨慎排查路由"降到"写一个新 handler 文件并注册"。

---

## 推荐方案：以 `IntentHandler` 协议为核心的插件化重构

核心思想：把目前散落在 `nodes.py / _react_internals.py / registry.py / cli.py` 中的 per-intent 分支，抽成实现统一协议的 handler 对象，集中放在新增的 `askanswer/intents/` 包内；`registry.py` 同步升级 tag 系统让工具不再绑死单一 intent。

### 阶段 1 · 类型紧致化（基础，吸收 TODO P0）

**目标**：让所有 intent / step 用 `Literal` 类型表达，扔掉手写 JSON 解析。

- `askanswer/state.py`
  - `IntentName = Literal["chat", "search", "sql", "file_read"]`
  - `StepName = Literal["understood", "tool_called", "completed", "retry_search"]`
  - `SearchState.intent: IntentName`，`step: StepName`
- `askanswer/nodes.py`
  - 用 `model.with_structured_output(IntentClassification)` 替换 `_parse_json_object` + `_intent_from_llm` 手写 JSON（吸收 TODO P0 第 2 条）
  - 定义 `class IntentClassification(BaseModel)`，字段 `intent: IntentName / file_path / search_query / understanding`，无需 markdown 围栏容错

**何处复用**：`_normalize_intent` 仍保留，作为 `IntentClassification.model_post_init` 的归一化逻辑。

### 阶段 2 · 抽出 `IntentHandler` 协议

**目标**：`nodes.py` 与 `_react_internals.py` 中所有"按 intent 分支"的逻辑收敛到一个对象上。

新建 `askanswer/intents/`：

```
intents/
  __init__.py          # IntentRegistry 单例 + register / get
  base.py              # IntentHandler protocol/dataclass + EvaluationResult
  chat.py              # ChatHandler
  search.py            # SearchHandler（含原 sorcery 评估逻辑）
  sql.py               # SqlHandler
  file_read.py         # FileReadHandler
```

`base.py` 定义：

```python
@dataclass(frozen=True)
class EvaluationResult:
    decision: Literal["pass", "retry"]
    retry_directive: dict | None = None  # {"search_query": "...", "file_path": "..."}
    reason: str = ""

class IntentHandler(Protocol):
    name: IntentName                     # 用于 registry.get
    bundle_tags: frozenset[str]          # 工具 tag 过滤集（不再硬编码 bundle 名）
    max_retries: int                     # 替代 nodes.py:351 的 `>= 1` 硬阈值

    def local_classify(self, text: str) -> IntentClassification | None: ...
    def prompt_hint(self, state: SearchState) -> str: ...   # 替 _answer_node 的 if/elif
    def evaluate(self, state: SearchState) -> EvaluationResult: ...  # 替 sorcery 主体
    def cli_label(self, update: dict) -> str: ...           # 替 cli.py:322-371 的分支
```

迁移：
- `_local_intent` 级联（`nodes.py:134-224`）拆成各 handler 的 `local_classify`，按注册顺序遍历调用
- `_answer_node` 中 `if intent == "chat": context_line = ...` 这一段（`_react_internals.py:62-77`）替换为 `intent_registry.get(intent).prompt_hint(state)`
- `sorcery_answer_node` 中"仅 search 评估、retry_count >= 1 放行"那一坨（`nodes.py:336-399`）变成 `handler.evaluate(state)`，返回 `EvaluationResult`，`graph.py` 的 `route_from_sorcery` 据此路由
- `_reclassify_intent`（`_react_internals.py:32-52`）改为遍历所有 handler 的 `local_classify`，第一个非 None 返回值胜出

### 阶段 3 · 把 sorcery 升级为可插拔评估（吸收 TODO P1 第 3、4 条）

**目标**：每个 intent 自带评估逻辑；摆脱"合成 HumanMessage"那种脆弱的字符串协议。

- `state.py` 新增 `retry_directive: dict` 字段（持久到 checkpointer），替代当前用合成 HumanMessage 传递新搜索词的做法
- `sorcery_answer_node` 简化为：
  ```python
  def sorcery_answer_node(state):
      handler = intent_registry.get(state["intent"])
      if state.get("retry_count", 0) >= handler.max_retries:
          return _finalize(state)
      result = handler.evaluate(state)
      if result.decision == "pass":
          return _finalize(state)
      return {
          "step": "retry_search",  # 沿用旧名以兼容现有路由
          "retry_count": state.get("retry_count", 0) + 1,
          "retry_directive": result.retry_directive,
      }
  ```
- `_answer_node` 在每轮检查 `state.get("retry_directive")`，如有则把它转译成 system prompt 增量（"上一次回答不够，请按以下指引重试：…"），不再注入合成 HumanMessage
- `SearchHandler.evaluate` 内置原 LLM 打分逻辑；`FileReadHandler.evaluate` 可检查 `read_file` 是否报错并改写路径；`ChatHandler.evaluate` 默认 always-pass；`SqlHandler.evaluate` 检查最近一条 ToolMessage 是否为空结果集
- `max_retries` 各 intent 独立：search=2（含指数退避去重）、file_read=1、chat=0、sql=1
- 去重：`retry_directive` 与历史 `search_query` 比较，相同则放弃重试（避免抖动）

### 阶段 4 · `ToolRegistry` 升级 tag 系统

**目标**：工具不再被一组固定 bundle 名约束；新增 intent 不必修改 registry 常量；HITL 类别可扩展。

`registry.py` 改动：
- `ToolDescriptor.bundles: frozenset[str]` → `tags: frozenset[str]`，bundle 名仍可用作 tag（向后兼容）
- 工具按多 tag 标签注册，例如 `tavily_search` → `{"chat","search","sql","file_read","external_api","io_bound"}`
- `ToolRegistry.list(intent=...)` 内部转为 `list(tags={intent})`，过滤逻辑不变
- `requires_confirmation: bool` → `confirmation_class: Literal["none","shell","fs_write","external_api_paid"]`
- `confirmation_names()` → `confirmation_classes()` 返回 `dict[str, str]`，react 路由按 class 决定走哪条 HITL 旁路（当前只接 shell，但门面已开）

`IntentHandler.bundle_tags` 在阶段 2 已加，自然衔接；新 intent 注册时附 `bundle_tags = frozenset({"my_intent"})`，工具按需贴 `my_intent` tag 即可被纳入。

### 阶段 5 · 子图拆分（吸收 TODO P1 第 2 条）

**目标**：`answer ⇄ tools / shell_plan` 这套循环独立成 `tool_loop_subgraph`，主图只看一个"生成回答"节点。

- 现有 `react.py:build_react_subgraph()` 已经是子图，但主图仍直接用它替换 answer 节点。把它显式重命名为 `tool_loop_subgraph`，并把 `_react_internals.py` 中跟 intent 相关的逻辑都搬到 handler 里
- 主图 `graph.py` 增加一个**条件入口边**：`understand → (sorcery_required ? tool_loop : tool_loop_no_eval)`，让 chat/file_read 默认走"无评估"分支，省掉一个无意义的 sorcery 节点跳转（评估 handler 内部已能 always-pass，但少一跳能去掉一次 checkpointer 写入）

### 阶段 6 · CLI 渲染插件化

**目标**：CLI 不再"知道"具体 intent 名。

- `cli.py:_render_node_update`（`cli.py:322-371`）的 understand/answer/sorcery 分支都改为 `intent_registry.get(update.get("intent","")).cli_label(update)` 取 detail 字符串
- `_marker(node, detail)` 调用不变；通用兜底分支保持原样
- 加 intent 时只需在 handler 里写 `cli_label`，CLI 文件不动

### 阶段 7 · 验收：加一个新 intent 作为冒烟

**目标**：用一个 < 80 LOC 的 `MathHandler`（数学题：识别"算 23*89"等表达式，调用 `calculate` 工具）验证整条扩展路径。

- 新文件 `intents/math.py` + 在 `intents/__init__.py` 里 `register(MathHandler())`
- 不动 `nodes.py / _react_internals.py / cli.py / registry.py / graph.py`
- 跑 `askanswer "算 17 * 23 + 4"` 走通即视为可扩展性达成

---

## 同时吸收的 TODO 项

| TODO 条目 | 阶段 | 说明 |
|---|---|---|
| P0 结构化输出 `with_structured_output` | 阶段 1 | 替手写 JSON 解析 |
| P0 `step`/`intent` Literal | 阶段 1 | 类型紧致化 |
| P1 sorcery 扩到 file_read/chat | 阶段 3 | `IntentHandler.evaluate` 即此项 |
| P1 多轮 retry 策略 | 阶段 3 | `max_retries` per intent + 去重 |
| P1 子图拆分 | 阶段 5 | `tool_loop_subgraph` 命名 + 主图条件分支 |

**不在本计划范围**（保留在 TODO，单独推进）：
- token 级流式（独立 CLI/streaming 工作，不影响图扩展性）
- `understand` 并行预取 `Send`（性能优化，不影响 handler 协议）
- LangSmith tracing（可观测性）
- 工具调用超时/缓存/MCP 健康检查（工具层而非编排层）
- 测试与文档（依赖本计划稳定后铺设）

---

## 关键改动文件

| 文件 | 改动 | 风险 |
|---|---|---|
| `askanswer/state.py` | 加 `IntentName/StepName` Literal、`retry_directive: dict` 字段 | 低，TypedDict 加字段向后兼容 |
| `askanswer/nodes.py` | `understand_query_node` 改用 IntentRegistry 遍历；sorcery 委托给 handler.evaluate | 中，需保证级联顺序与现行行为一致 |
| `askanswer/_react_internals.py` | `_answer_node` 用 `handler.prompt_hint`；`_reclassify_intent` 走 registry | 中，prompt 内容必须等价 |
| `askanswer/registry.py` | `bundles → tags`；`requires_confirmation → confirmation_class` | 中，所有 `_seed_*` 注册点同步迁移 |
| `askanswer/graph.py` | `route_from_sorcery` 据 `EvaluationResult.decision` 路由；可选条件入口边 | 低 |
| `askanswer/cli.py` | `_render_node_update` 的 intent 分支替成 `handler.cli_label` | 低 |
| `askanswer/react.py` | 重命名 build_react_subgraph → build_tool_loop_subgraph | 低（语义重命名） |
| 新增 `askanswer/intents/` | `__init__.py / base.py / chat.py / search.py / sql.py / file_read.py` | — |

---

## 复用现有代码

- `_local_intent`（`nodes.py:134-224`）的关键词表与正则按 intent 切片到各 handler 的 `local_classify` 实现，**逻辑保持原样**，只是搬家
- `_normalize_intent`（`nodes.py:248-272`）作为 `IntentClassification` 的归一化辅助函数复用
- `ToolRegistry` 单例骨架（`registry.py:53-115`）保留，仅把 `bundles` 字段语义放宽为 `tags`
- `_run_with_confirmation`（`_react_internals.py:239`）保留，仅按 `confirmation_class` 选执行体
- `sorcery_answer_node` 中 LLM 评分 prompt（`nodes.py:359-372`）整体迁到 `SearchHandler.evaluate`，提示词内容不变
- 持久化层（`persistence.py`）零改动，新增的 `retry_directive` 字段由 `add_messages` 不接管，但 SearchState 是 TypedDict，checkpointer 自动序列化

---

## 验证

无测试套件，按以下顺序手动验证：

1. **类型/启动**：`python -c "from askanswer.graph import create_search_assistant; create_search_assistant()"` 不报错；`askanswer --graph` 输出的 Mermaid 与重构前对比，节点名/边可解释
2. **四类 intent 冒烟**（每条单跑一次）：
   - `askanswer "你好"`（chat：不走 sorcery）
   - `askanswer "搜一下今日 BTC 价格"`（search：走 sorcery，可能触发 retry）
   - `askanswer "读 ./README.md 总结一下"`（file_read：file_read handler.evaluate 处理读失败重试）
   - `askanswer "查一下 user 表结构"`（sql：sql_query 工具调用走通）
3. **HITL 回归**：`askanswer "帮我列出当前目录大文件"` → shell 确认弹出 → y → 命令执行；测 e（编辑指令）路径
4. **重分类回归**：REPL 内先 chat 一句，再问 SQL 问题，确认中途切换 intent 仍能命中正确工具集
5. **新 intent 接入测**：阶段 7 的 `MathHandler` 走通即视为可扩展性达成
6. **MCP 重连**：`/mcp <url>` → `/mcp tools` 列出新工具 → 提一条用得上的问题，确认 LLM 能调用

仅当所有冒烟通过、Mermaid 可解释、阶段 7 新 intent 接入成功，才视为重构完成。

---

## 可行性评估（2026-05-06 review）

### ✅ 站得住的部分
- **行号引用全部对齐**：`nodes.py:14-122/275-294/336-399`、`_react_internals.py:32-52/62-77`、`registry.py:26-50`、`cli.py:322-371`、`state.py` 字段都已核对，与现实一致。
- **`with_structured_output` 替换手写 JSON 解析**：`load.py` 是 `init_chat_model` 代理，OpenAI/Anthropic 都原生支持，零阻碍。
- **`bundles → tags` 重命名**：`registry.list(bundle=...)` 只在 `_react_internals.py:80` 一处使用，迁移面小、可保留 `bundle` 关键字别名向后兼容。
- **`retry_directive` 走 TypedDict**：`SearchState` 加可选字段对 SqliteSaver 透明。
- **react 子图改名**为 `tool_loop_subgraph`：纯重命名，零风险。

### ⚠️ 三处必须先解决的设计冲突

**1. `IntentName = Literal[...]`（阶段 1）与"插件化注册新 intent"（阶段 2、7）冲突**
Literal 是闭合的，运行时无法扩展。新 `MathHandler` 一加，`state["intent"]` 静态类型立即不合法。要么：
- 退回 `intent: str` + handler.name 校验（放弃 P0 Literal），或
- 维持 Literal 但接受"插件 intent 类型上不安全"。
计划需声明取舍。

**2. `local_classify` 第一个非 None 胜出 vs 现有优先级语义**
当前 `_local_intent`（`nodes.py:144-178`）有强排序：file_path+动作词 > SQL 关键词/正则 > 搜索关键词 > chat 起手词，且 fallback 路径里"长文本无问号 → search"是**跨 intent 的全局启发**。拆到各 handler 的话：
- 注册顺序就是分类优先级，新加 intent 易踩坑；
- 全局 fallback（长度阈值、空文本）无主，应该放在 registry 协调层、不属于任何单个 handler。
计划只说"按注册顺序遍历"，未说优先级如何固定。

**3. `chat` handler 也接 sorcery 评估（阶段 3）但 `max_retries=0`**
阶段 5 又提让 chat/file_read"走 tool_loop_no_eval 分支跳过 sorcery"。两者重复——`max_retries=0` 已经能 always-pass，多搞一条入口边纯粹给 `graph.py` 加复杂度。**建议砍掉阶段 5 的条件入口边**，保持单一入口。

### 🔸 次要提示
- `_normalize_intent` 用 `model_post_init` 不太合适（pydantic 签名约束），改 `@model_validator(mode="after")` 更顺。
- `IntentHandler.cli_label(update: dict)` 把 UI 文案塞进 `intents/` 包，跨层耦合可接受但要意识到。
- 阶段 7 用 `MathHandler` 做冒烟较弱：`calculate` 工具已在所有 bundle，chat handler 直接覆盖；建议换成"加 `code_analysis` intent，挂 `read_file + tavily_search` 两个 tag"才能真正证明 tag 系统起作用。
- 没有测试套件，6 阶段连改 7 文件，回归靠手测——建议在阶段 1 之前先把 `验证` 一节的 6 条冒烟脚本化（一个 `make smoke` 脚本调 4 条 `askanswer` 命令断言关键词），再开干。

### 推荐顺序微调
阶段 1 → 阶段 4（tag 改名，独立可验证）→ 阶段 2/3（handler 抽取，最大改动一次到位）→ 阶段 6（CLI）→ 阶段 7（冒烟）。砍掉阶段 5 的条件入口边。

