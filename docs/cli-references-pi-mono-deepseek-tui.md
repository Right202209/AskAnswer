# CLI/TUI 参考对比:pi-mono vs DeepSeek-TUI vs AskAnswer

调研日期:2026-05-07
参考仓库:
- [badlogic/pi-mono](https://github.com/badlogic/pi-mono) — TS monorepo,含 `pi-tui` 库 + `pi-coding-agent` CLI
- [Hmbown/DeepSeek-TUI](https://github.com/Hmbown/DeepSeek-TUI) — Rust + ratatui,完整化的 DeepSeek 终端 agent

目标:为 AskAnswer 的 CLI/REPL 改造提供架构参考与可提取功能清单。

---

## 一、三者基本面

| 维度 | **pi-mono** | **DeepSeek-TUI** | **AskAnswer (现状)** |
|---|---|---|---|
| 语言/栈 | TS monorepo, Node | Rust workspace, ratatui | Python, LangGraph |
| 形态 | 5 个 npm 包(库 + CLI + Web) | 多 crate 单产品(`deepseek` + `deepseek-tui`) | 单包 `askanswer` |
| TUI 实现 | 自研 `pi-tui`:差分渲染 + CSI 2026 + Component | 基于 `ratatui`(crossterm),App/UI/Approval 等组件 | 无 TUI 库,纯 ANSI 拼字符串 + `print` |
| Agent 引擎 | `pi-agent-core`:事件流式,转换层 `convertToLlm` | `core/engine` + `turn_loop` + `capacity_flow` | LangGraph `StateGraph` + react 子图 |
| 工具系统 | `AgentTool` + parallel/sequential + before/after hooks | 类型化 registry,3 种模式(Plan/Agent/YOLO) | `ToolRegistry` 标签制 + `confirmation_class` |
| 会话存储 | JSONL 树形(单文件多分支) | SQLite + 检查点 + 离线队列 + 侧 git 快照 | SQLite(checkpoint + thread_meta) |
| 多提供商 | ~25 家(订阅 + API key + OAuth) | DeepSeek + 任何 OpenAI 兼容端点 | 通过 `init_chat_model` |
| 扩展机制 | Extensions(TS) + Skills + Prompt templates + Themes + npm 包分发 | Skills(GitHub) + MCP + Hooks(jsonl/webhook) | MCP + 内置 IntentHandler 插件 |

---

## 二、核心设计理念差异

**pi-mono**:把"TUI 组件库"和"agent 内核"分别做成独立 npm 包,任何人都能搭自己的 CLI。亮点是**事件流模型**(`agent_start/turn_*/message_update/tool_execution_*/agent_end`)和**消息队列**(steering=插队,follow-up=排队)。

**DeepSeek-TUI**:走"完整 IDE 化 agent"路线 — Plan/Agent/YOLO 三模式 + 侧 git 快照回滚 + LSP 诊断回灌 + 持久化任务队列 + HTTP/SSE 服务化。是把 Claude Code 那套架构在 Rust + DeepSeek 上重做了一遍。

**AskAnswer**:focus 在 LangGraph 的"intent 路由 + 多意图处理器"抽象上,TUI 和 agent 体验弱;HITL 仅 shell,没有 LSP/快照/任务队列等。

---

## 三、可提取功能清单(按 ROI 排序)

### A. 直接能落地的小项(对现有 cli.py / persistence.py 增量改)

1. **`!cmd` 不发送 / `!!cmd` 发送**(pi)— 区分"跑 shell 但不喂给 LLM"和"跑 + 把输出贴回上下文"。AskAnswer 现在只有前者。
2. **消息队列 / 插队**(pi)— Enter 排队"steering"消息,等当前 turn 完成后注入;Alt+Enter 排队"follow-up"。`stream_query` 改造成生产/消费即可。
3. **离线/失败队列持久化**(DeepSeek)— `pending_shell` 已有思路,扩到所有用户输入:写到 `~/.askanswer/checkpoints/offline_queue.json`,启动时回放。
4. **`/fork <thread_id>` + `/branch`**(pi `/tree`、DeepSeek `deepseek fork`)— `thread_meta` 已有 thread 概念,加一个 `parent_thread_id` 字段就能做"从某个用户消息分叉"。
5. **`/export` / `/share`**(pi)— 导出整个 thread 为 HTML / Markdown,或上传 gist。`thread_meta` + checkpoints 直接序列化即可。
6. **Shift+Tab 循环 thinking-level**(DeepSeek/pi)— 现在 `/model` 只切模型,加一个 reasoning effort 的快捷循环。
7. **`@文件` 模糊搜索补全**(pi)— Editor 里 `@` 触发,直接和现有 `read_file` 工具串起来。需要 fuzzy 库或自己写最长子序列。

### B. 中等改造(影响一两个模块)

8. **流式输出 + 事件总线**(pi `agent_event` 模型)— 现状是 `app.stream(stream_mode="updates")` 节点级;升级到 token 级 + `tool_execution_update`(`stream_mode="messages"`)。已是 `TODO.md` P1。
9. **执行策略层**(DeepSeek `crates/execpolicy`)— 把"该不该 ask 用户"从 `confirmation_class` 升级成可配策略文件:不同工具/路径前缀/正则不同动作。
10. **Hooks 系统**(DeepSeek/pi)— 工具前后挂 `command/webhook/jsonl`。和 `_tools_node` 的工具调用点对接即可。
11. **Cost tracking**(DeepSeek)— per-turn token + 估价存进 `thread_meta`,`/status` 显示。LangChain 的 `usage_metadata` 已有数据,差最后一公里。
12. **审计日志**(DeepSeek `~/.deepseek/audit.log`)— 凭据变更、shell 批准、模型切换等只追加。`persistence.py` 加一张 `audit` 表即可。
13. **Skills/Prompt templates**(pi/DeepSeek)— 把"指令包"放 `~/.askanswer/skills/<name>/SKILL.md`,作为 `/skill:name` 注入 system prompt。比 IntentHandler 轻量。

### C. 大件(架构级,要先权衡)

14. **TUI 重写**(pi-tui 差分渲染 / DeepSeek ratatui)— 当前 cli.py 1091 行手搓 ANSI,边界、IME、resize、overlay 都欠。Python 侧候选:`textual`(对标 pi-tui)或 `prompt_toolkit` + `rich.live`(渐进迁移)。
15. **侧 git 快照 + `/restore`**(DeepSeek)— 在 `~/.askanswer/snapshots/<repo_hash>/.git` 做 pre/post-turn snapshot,不动用户的 `.git`。对 file_read/SQL 收益小,对未来加 fs_write 工具是必需。
16. **Auto-mode 路由**(DeepSeek)— 先用 cheap 模型做 1 次 routing call 决定真正这一轮用什么模型 + 推理强度。和 `understand_query_node` 已有的 LLM 分类合并成一步。
17. **HTTP/SSE 运行时**(DeepSeek `serve --http` + ACP)— 把 LangGraph 应用裸暴露成 HTTP/SSE,让 Zed/IDE 用 ACP 接入。
18. **持久化任务队列**(DeepSeek `task_manager`)— 长任务(深度调研、批量 SQL)放队列异步跑,worker pool 限并发,任务跨重启存活。

### D. 不建议直接抄

- pi 的 **Pi Packages**(npm 分发扩展)— AskAnswer 是 Python 包,生态对不上;PyPI 风格的扩展点更合适。
- DeepSeek 的 **LSP 诊断回灌** — AskAnswer 现在工具集没有 fs_write,LSP 投入产出比低,等真正做编码 agent 再说。
- DeepSeek 的 **macOS Seatbelt 沙箱** — 只在 shell 工具有副作用,且已有 HITL 拦截,沙箱性价比低。

---

## 四、推荐起步顺序

如果只挑 3 件先做(渐进、低风险):

1. **A2 消息队列** — 立刻提升交互流畅度,不动 LangGraph 内核。
2. **B8 token 级流式** + **B11 cost tracking** — 一次性把 stream 模式升级到 `messages`,顺手把 usage 落库。
3. **A4 `/fork`** + **A5 `/export`** — 拓展 thread 价值,撑起 `enterprise-persistence-plan.md` 的 phase B。

---

## 五、关键文件位置参考(便于深入挖)

**pi-mono** (TS):
- `packages/tui/src/{tui.ts,editor-component.ts,components/}` — 差分渲染 TUI 库
- `packages/agent/` — 事件流 agent 内核(`convertToLlm`、`transformContext`、`beforeToolCall`)
- `packages/coding-agent/` — 完整 CLI(session JSONL 树形分支、`/tree`、`/fork`、`/export`)

**DeepSeek-TUI** (Rust):
- `crates/tui/src/{app.rs,ui.rs,approval.rs,streaming.rs}` — ratatui 渲染
- `crates/core/src/engine/{turn_loop.rs,capacity_flow.rs}` — agent 主循环
- `crates/state/` — SQLite 持久化
- `crates/execpolicy/` — 工具批准/沙箱策略
- `crates/tui/src/lsp/` — LSP 诊断子系统
- `docs/ARCHITECTURE.md` — 全景架构文档

**AskAnswer 待改造点**:
- `askanswer/cli.py:1-1091` — REPL 主循环、ANSI 自绘、`stream_query`
- `askanswer/persistence.py` — `thread_meta` 表(待加 `parent_thread_id`、`cost`、`audit`)
- `askanswer/_react_internals.py` — `_tools_node` 是 hooks 接入点
- `askanswer/registry.py` — `confirmation_class` 已具雏形,可升级到 execpolicy
