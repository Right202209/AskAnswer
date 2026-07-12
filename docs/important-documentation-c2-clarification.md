# Important Documentation — C2 通用澄清能力（generic clarification）

> 状态：**代码已落地，尚未运行验证**（本轮遵循「只写代码、不跑 Python」）。
> 本文记录改动与设计不变量；**逐项验证清单与提交闸门已并入**
> `docs/important-documentation-verification-matrix.md`（总表）。
> 对应 `plan-docs/02-execution-plan.md` 的 **C2**；该进度框在总表对应闸门通过前**不勾选**。

## 1. 改动清单

| 文件 | 改动 |
|---|---|
| `askanswer/intents/base.py` | 新增 `ClarificationChoice`、`ClarificationRequest` 数据类；新增 `get_clarification(handler, state, context)` 安全分发助手；在 `IntentHandler` 协议注释里声明可选方法 `clarify` |
| `askanswer/intents/file_read.py` | 新增 `clarify`：无路径时给「跳过 / 手动输入路径」 |
| `askanswer/intents/sql.py` | 新增 `clarify`：`context.db_dsn` 缺失时给「仍按库处理 / 改用通用知识(intent→chat)」 |
| `askanswer/intents/research.py` | 新增 `clarify` + 常量 `RESEARCH_SCOPE_MIN_CHARS=24`：主题过短时给 4 个聚焦角度 |
| `askanswer/clarify.py`（新） | react 子图入口节点 `clarify_node` + 载荷/解析助手 |
| `askanswer/react.py` | 装入 `clarify` 节点，入口改为 `START → clarify → answer` |
| `askanswer/cli.py` | `_prompt_confirmation` 增加 `type=="clarify"` 分发；新增 `_prompt_clarification`；`_PHASE_TEXT`/`_render_node_update`/`_clarify_detail` 进度渲染；`_on_node_update` 拦截空澄清；import 增 `is_interactive`、删未用的 `CANCELLED` |

**未改动**：`state.py`（无需新字段）、`nodes.py`、`_react_internals.py`、父图 `graph.py` 拓扑。

## 2. 设计不变量（验证时重点确认）

1. **父图拓扑不变**：`START → understand → answer → sorcery → (END|answer)`。clarify 只在 `answer`（react 子图）**内部**，故 `01-spec.md` 不变量 1 成立。
2. **每轮最多澄清一次**：`clarify_node` 用 `state["step"] == "understood"` 守卫。sorcery 重试（`step=retry_search`）与 react 内部 `answer⇄tools` 循环都不会重入 clarify。
3. **不重跑 LLM**：intent 早在 understand 阶段落库；clarify_node 只读已提交 state + 运行时 context，interrupt 恢复后重算廉价且确定，**不触发 `_intent_from_llm`**。
4. **默认项 = 保持现状**：每个 `ClarificationRequest.choices[default_index]` 的 `updates` 为空/等价现状；非 TTY（`is_interactive()` 假）直接取默认 → **非交互路径零行为变化**。
5. **失败降级**：`get_clarification` 吞掉 handler 异常返回 None；澄清永不阻断回答。

## 3. 交互/恢复时序（与 confirmations 同构）

```
understand(step=understood) → [answer 子图] START → clarify_node
  clarify_node: request = handler.clarify(state, ctx)
    request is None → return {} → answer          # 绝大多数轮次
    request 非空   → interrupt({"type":"clarify", labels,...})
      ↑ 透传父 stream → CLI stream_query 捕获 __interrupt__
        → _prompt_confirmation → _prompt_clarification
            TTY   : select_option 菜单 → {"index","text"}
            非 TTY: 直接 {"index":default_index,"text":None}
        → Command(resume=...) 续跑
      ↓ clarify_node 重入，interrupt() 返回 resume 值
    _resolve(request, decision) → 并回 SearchState 的部分字段 → answer
```

- resume 值形态 `{"index": int, "text": str|None}`；`_resolve` 里手动输入项 index == `len(choices)`。
- `index == -1`（Esc/CANCELLED）或越界 → 返回 `{}`（保持现状）。

## 4. 必测清单

> **已迁移**：本节原逐项清单已并入总表
> `docs/important-documentation-verification-matrix.md`（条目编号 R1/R3/R4/R5 + C2-U1..U6 + C2-G1..G5 + C2-R1..R4），
> 执行与勾选一律以总表为准，本节不再维护逐项框。原 4.1 冒烟 → 总表 G0；4.2 单元 → G2；4.3 节点/图 → G3；4.4 非回归 → G4。

## 5. 已知取舍 / 待办
- file_read 目前只提供「跳过 + 手动输入路径」，未做 CWD 候选扫描（避免 interrupt 重入时的非确定性与误读文件）。若要加候选，需保证扫描确定性（`sorted`）或把 request 持久化到 state。
- `RESEARCH_SCOPE_MIN_CHARS=24` 为启发式阈值，可按实测调整；默认项为「全面概览」，误触发也无害。
- 提交前仍需：过安全清单、更新 `CHANGELOG.md`、通过总表矩阵的 G0+G2+G3+G4 后再勾选 `02-execution-plan.md` 的 C2（提交闸门详见总表 §4，含 cli.py 按 C2/C3 拆 hunk 的说明）。
