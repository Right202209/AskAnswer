# Claude Code 架构学习笔记

学习日期:2026-05-07
来源:Claude Code 内部架构 Mermaid 流程图
目的:为 AskAnswer 的编排/工具/扩展层设计提供参考。

---

## 一、整体分层

Claude Code 把运行时切成 **5 层**,层间只通过明确接口交互:

| 层 | 职责 | 对应 AskAnswer 的位置 |
|---|---|---|
| **Entry** | `cli.js` → `main.tsx`,启动 init/settings/policy/telemetry | `cli.py` + `__main__.py` |
| **Registry** | 命令、工具、Skills、Agents、MCP 五张注册表的 bootstrap | `registry.py`(Tool) + `intents/__init__.py`(Intent) |
| **Runtime** | REPL 与 SDK QueryEngine 共享同一 `query()` 状态循环 | `graph.py` + `cli.py:repl()` |
| **Tool Orchestration** | LangGraph 节点:分区→校验→权限→执行→后置 | `tools.py` 的 `react_subgraph` |
| **State & Observability** | transcript / memory / cache / 压缩 / 追踪 | SQLite checkpoint + `thread_meta` |

```
CLI → main → {init, registry, runtime}
registry: Commands | Tools | Skills | Agents | MCP
runtime:  REPL ↘
                query() loop ⇄ Model
          SDK  ↗     ↓
                   tool_use? → Tool Orchestration → tool_result → 回到 query()
```

---

## 二、关键设计观察

### 1. Registry 五合一,但 **MCP 与本地工具走同一个 `findTool`**
`Tool Registry` 既挂本地核心工具(File/Bash/Grep/...),也挂 conditional 工具(REPL/LSP/Browser/Cron),还挂 `MCPTool` 包装器。MCP 客户端独立,但出口收敛到统一的工具调用接口。

> **AskAnswer 当前已经做到一半**:`ToolRegistry` 用 tag 区分,但 MCP 工具是另外一条路径。可考虑统一成"所有工具都走 registry,MCP 只是来源标签"。

### 2. Tool Orchestration 是一个 LangGraph 节点,内部又是流水线

```
partition(parallel/serial) → runToolUse → findTool → Zod 校验 → 权限引擎
   → pre-hook → approval(if needed) → allow? → tool.call() → post-hook → tool_result
```

要点:
- **partition 在前**:并行安全(只读 Read/Grep/Glob)与串行不安全(Edit/Write/Bash)分开调度。
- **权限引擎与 hooks 分离**:hook 是用户/项目可注入的回调,permission 是系统级策略。
- **Zod 校验在权限之前**:防止恶意参数绕过 permission 判断。
- **拒绝路径也产生 `tool_result`**:模型必须看到"被拒"反馈,而不是静默丢弃。

### 3. 子调用(Agent / Skill / MCP)递归回到同一个 `query()` 循环

```
EXEC → AGENT_CALL → runAgent() → sub-agent LangGraph loop → query()
```

子 agent 不是新进程,而是 **同一个 query() 状态机的另一个实例**,共享 transcript/memory 协议。这意味着:
- 主循环和子循环可以共用工具/权限/hooks。
- 任何工具都可以"升级"成 sub-agent(给它一个独立 context window 即可)。

### 4. 状态层三件套并列

`query()` 同时输出到三处:
- **state**:transcript / memory / cache(对话历史 + 短期缓存)
- **observability**:telemetry / tracing / cost
- **memory**:context compaction / 跨会话记忆

`STATE → RUNTIME` 反馈是为了 REPL 能恢复;`MEMORY → QUERY` 反馈是为了压缩后再注入。

---

## 三、可借鉴到 AskAnswer 的点(按 ROI 排序)

### A. 即可落地

1. **工具调用前置 partition**:目前 react 子图的并行能力依赖 LangGraph 默认调度,可显式区分 `parallel_safe` / `serial_only` 标签(只读类工具自动并行)。
2. **拒绝路径也回写 ToolMessage**:确认现在 `confirmation_class` 拒绝时是否产生 `tool_result`,否则模型可能"以为工具没跑"而重试。
3. **Zod-style 参数校验提到权限之前**:用 pydantic 在 IntentHandler 入口先校验,统一错误信息格式。

### B. 中期演进

4. **Sub-agent 共享 query 循环**:目前 SQL 子图和 Helix 子图各自独立,可以抽出"通用 sub-graph runner",让 IntentHandler 通过它声明性地启动子任务。
5. **Hooks 与 Permission 解耦**:当前 `confirmation_class` 既是策略又是钩子,拆成 `policy`(系统级)+ `hook`(用户级)两层。

### C. 长期(架构级)

6. **统一 Registry**:Intent / Tool / MCP / Skill 合并成一张表,以 `kind` 字段区分。Claude Code 的设计证明这种统一是可行且更简洁的。
7. **observability 第一公民**:目前 `monitoring-plan.md` 还在规划阶段,Claude Code 把 telemetry 与 state 并列写在主循环出口,值得参照。

---

## 四、与已有调研的关系

- **vs `cli-references-pi-mono-deepseek-tui.md`**:pi-mono 的"事件流"和 DeepSeek-TUI 的"Plan/Agent/YOLO 三模式"对应 Claude Code 这里的 `permission engine + approval`;Claude Code 没有显式三模式,而是用 permission policy 配置出来。
- **vs `orchestration-extensibility-plan.md`**:该文档已规划 Intent/Tool 注册表抽象,本笔记进一步指出"应当让 sub-agent 也走同一注册表"。
- **vs `helix-subgraph-plan.md`**:Helix 子图就是 Claude Code "sub-agent loop" 的一个具体实例,可以用同一抽象统一。
