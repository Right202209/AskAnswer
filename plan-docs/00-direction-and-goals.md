# AskAnswer 项目方向与目标（2026-07 重规划）

> 本文取代 `docs/possibility-space-*.md` 作为当前方向的权威文档。旧计划的方向卡片仍有参考价值，但其执行部分已基本完成，不再是行动依据。

## 1. 现状诊断（重规划的动因）

1. **功能跑在验证前面**。旧计划 Phase 0–3.4 已全部实现：MCP 健康检查与 profile 自动重连、tenant 隔离（schema v3）、`/undo` 标签（schema v4）、telemetry 双导出器、`research_brief` 与 `decision_memo` 子图、Helix MCP server。其中 1.2–3.4 约 2000 行改动**未提交**，仅经 `py_compile` 验证。
2. **零测试**。仓库没有 `tests/`、没有 linter、没有 CI。两次 schema 迁移（v2→v3→v4）、确认协议泛化、多租户过滤这些高风险面完全没有回归保护。
3. **模块超限**。`cli.py` 2466 行、`persistence.py` 875 行、`tools.py` 581 行、`mcp.py` 547 行，远超 300 行/文件的硬性质量上限；后续任何功能都要触达这些文件，越晚拆越贵。

## 2. 新方向：由广度转深度

**先落地（Land）→ 再加固（Harden）→ 后演进（Evolve）。**
在测试与质量门槛就位之前，不再开新功能面。

### G1 · Land — 安全落地在途工作

把工作区里的 7 项功能变成可信的提交：运行时冒烟验证（超越 `py_compile`）、安全检查清单、按逻辑单元提交。完成标志：`git status` 干净，CHANGELOG 与提交一一对应。

### G2 · Harden — 建立安全网

pytest 骨架 + 针对最高风险面的测试（迁移、tenant 过滤、UPSERT、确认协议、mcp_profile 原子写、telemetry 门控、handler 注册）；接入 ruff；测试命令写回 `.claude/mem/commands.md`。完成标志：`pytest` 与 `ruff check` 一条命令可跑、全绿。

### G3 · Evolve — 在坚实底座上演进形态

旧计划 Phase 4 的重新排序版，且以 **拆分 cli.py** 为第一步（它同时是 HTTP server 的前置和 300 行上限的欠账）：

1. C1 拆 `cli.py`：runner（事件流）与 renderer（终端 UI）分离，命令处理拆包。
2. C2 通用澄清能力（`ClarificationRequest` 协议）。
3. C3 HTTP/SSE server（复用 C1 的 runner）。
4. C4 只读 Web UI（threads / audit / checkpoints 浏览，按 tenant 过滤）。

G3 只有在 G1、G2 全部完成后才启动。

## 3. 非目标（继承旧计划的放弃清单）

- `code_review` / `data_profile` 子图（低价值，无限期延后）
- Redis / 消息队列；完整 RBAC/SSO 企业管理台
- 按 intent 分叉父图拓扑；绕过 `ToolRegistry` 直接 `bind_tools`
- 向 `SearchState` 塞 telemetry/metrics；给 SQL agent 开写权限
- 把 `tenant_id` 写进 message 内容

## 4. 成功标准

| 目标 | 可验证标准 |
|---|---|
| G1 | 工作区干净；每个提交通过安全清单；`askanswer --graph` 与冒烟矩阵全过 |
| G2 | `pytest` ≥ 25 个用例覆盖 §G2 列出的风险面；`ruff check askanswer` 零报错 |
| G3 | C1 后 `cli.py` 主文件 ≤ 300 行；C3 后 CLI 与 HTTP 消费同一事件流；行为回归由 G2 测试保护 |

配套文档：`01-spec.md`（系统规格与不变量）、`02-execution-plan.md`（分步执行手册）。
