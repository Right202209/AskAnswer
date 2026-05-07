# Helix 子图：规格优先演化循环

> 把 `ouroboros/` 里的 Interview → Seed → Execute → Evaluate 演化循环按本仓库 SQL agent 的范式包成一个**可被 LLM 调用的工具子图**。

## 1. 设计目标

- 演化循环只在「明确想做规格化开发」的请求中触发，不污染父图拓扑。
- 子图自带递归与硬上限，对应 ouroboros 的多代演化语义。
- 入口由 LLM 自主判断，统一通过 react 节点的 tool-call 调用。

## 2. 文件布局（参照 `askanswer/sqlagent/`）

```
askanswer/helix/
├── __init__.py        # 空包标记
├── state.py           # HelixState（MessagesState + 演化字段）
├── nodes.py           # interview / seed / execute / evaluate 节点体
├── agent.py           # build_helix_agent() + run_helix_agent()
└── helix_tool.py      # @tool helix_spec_loop（封装为 LLM 可调用工具）
```

新增 intent：

```
askanswer/intents/helix.py        # HelixHandler
```

修改：

```
askanswer/intents/__init__.py     # 注册 HelixHandler
askanswer/registry.py             # 增加 TAG_HELIX、_seed_helix()
```

## 3. 子图状态（`state.py`）

```python
class HelixState(MessagesState):
    topic: str                      # 用户最初表述
    interview_qa: list[dict]        # [{"q": ..., "a": ...}]
    seed: dict                      # {goal, constraints, acceptance_criteria, ontology, principles}
    artifact: str                   # execute 阶段产出
    evaluation: dict                # {verdict, score, gaps[]}
    generation: int                 # 当前代数（从 1 起）
    lineage: list[dict]             # 历代 seed 快照
    verdict: str                    # "approved" | "rejected" | "exhausted"
```

## 4. 节点要点（`nodes.py`）

每个节点都用 `model.with_structured_output(<Pydantic>)`，避免再做正则。

- **interview_node**：模型按 Socratic 4 轨道（scope/constraints/outputs/verification）生成关键问题并自答（工具内无法真做 HITL）。
- **seed_node**：基于 `interview_qa`（首代）或 `evaluation.gaps`（后续代）输出 Seed Pydantic；快照写入 `lineage`。
- **execute_node**：根据 `seed.acceptance_criteria` 生成方案文本（不真跑代码）。
- **evaluate_node**：覆盖率自查 + 语义评分（0–1）+ gaps 列表。
- **router**（条件边）：
  - `evaluation.verdict == "approved"` → END
  - `generation >= MAX_GENERATIONS` → `verdict="exhausted"` → END
  - 否则 `generation += 1` → 跳回 `seed`

硬上限：

```python
MAX_GENERATIONS = 3
RECURSION_LIMIT = 24
```

## 5. 拓扑（`agent.py`）

```
START → interview → seed → execute → evaluate → router
                     ↑                              │
                     └──────── retry ───────────────┘
                                                    │
                                                  END
```

## 6. 工具包装（`helix_tool.py`）

```python
@tool
def helix_spec_loop(topic: str, runtime: ToolRuntime[ContextSchema]) -> str:
    """规格优先开发循环：苏格拉底澄清 → 生成 Seed → 产出方案 → 评估并迭代。
    适用：用户给出模糊需求/希望从需求层面演化规格的场景。
    返回：最终 Seed + 工件 + 评估摘要的 Markdown 文本。
    """
```

返回 Markdown：`## Goal / ## Constraints / ## Acceptance criteria / ## Artifact / ## Evaluation / ## Lineage`。

## 7. Intent（`intents/helix.py`）

```python
class HelixHandler:
    name = "helix"
    priority = 22
    bundle_tags = frozenset({"helix"})
    max_retries = 0
```

`local_classify` 关键字（中英）：

- 中文：苏格拉底、需求澄清、规格化、演化循环、澄清需求、生成 seed、crystallize
- 英文：helix、ouroboros、spec-first、specification first、socratic、interview me、acceptance criteria、evolve loop

`prompt_hint` 引导 LLM 调用 `helix_spec_loop`；`evaluate` 直接 `pass_result()`。

## 8. registry 改动

```python
TAG_HELIX = "helix"
ALL_INTENT_TAGS = frozenset({..., TAG_HELIX})

def _seed_helix(registry):
    from .helix.helix_tool import helix_spec_loop
    registry.register(ToolDescriptor(
        tool=helix_spec_loop,
        tags=frozenset({TAG_CHAT, TAG_HELIX, "helix_tool"}),
        source="helix",
    ))
```

`get_registry()` 中在 `_seed_sql(r)` 之后调用 `_seed_helix(r)`。

## 9. 验证

```bash
python -m compileall askanswer/helix askanswer/intents/helix.py askanswer/registry.py askanswer/intents/__init__.py
askanswer --graph /tmp/helix-check.mmd
askanswer "interview me about a CLI todo app"
```

## 10. 不做的事

- 不接 ouroboros 上游真实 MCP server。
- 不做 HITL（子图在工具内部跑，无法 `interrupt()`）。
- 不动父图拓扑、不改 `state.py`。

---

## Implementation status

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1 | 计划评审定稿 | done | 名字定为 `helix` |
| 2 | `askanswer/helix/state.py` | done | MessagesState + 演化字段 |
| 3 | `askanswer/helix/nodes.py` | done | interview/seed/execute/evaluate + route + finalize |
| 4 | `askanswer/helix/agent.py` | done | build/compile + run + format_summary |
| 5 | `askanswer/helix/helix_tool.py` | done | @tool helix_spec_loop |
| 6 | `askanswer/intents/helix.py` | done | HelixHandler with CN+EN keywords |
| 7 | `intents/__init__.py` 注册 | done | HelixHandler 加入 _registry |
| 8 | `registry.py` 增加 `TAG_HELIX` 与 seeding | done | _seed_helix() 紧跟 _seed_sql |
| 9 | `compileall` smoke check | done | 全包通过；handler/registry/tool 实例化均验证 |

## Verification log (2026-05-07)

```
$ python -m compileall -q askanswer            # OK
$ python -c "from askanswer.helix.agent import helix_agent; ..."
helix subgraph nodes: ['__end__', '__start__', 'evaluate', 'execute', 'finalize', 'interview', 'seed']
edges: __start__→interview→seed→execute→evaluate→(seed | finalize)→__end__
$ python -c "from askanswer.intents import get_intent_registry; ..."
handler names: ['chat', 'file_read', 'helix', 'math', 'search', 'sql']
TAG_HELIX in ALL_INTENT_TAGS: True
classify('用苏格拉底问我关于一个 CLI todo app') -> helix
classify('spec-first design for X')              -> helix
classify('normal weather query')                 -> None
$ python -c "from askanswer.registry import get_registry; ..."
helix_spec_loop registered: True
source=helix tags=['chat','helix','helix_tool'] confirm=none
```

父图 `--graph` 仍正常输出 understand→answer→sorcery 拓扑，未受影响。
