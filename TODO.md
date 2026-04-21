# AskAnswer TODO

基于当前 LangGraph 实现（`graph.py` / `nodes.py` / `state.py`）梳理的改进清单，按优先级排列。

## 🔥 P0 · 基础设施（为后续所有改进铺路）

- [ ] **持久化 checkpointer**：把 `InMemorySaver` 换成 `SqliteSaver`（本地文件 `~/.askanswer/state.db`），实现跨进程会话恢复。`graph.py:73`
- [ ] **结构化输出**：`understand_query_node` 里的 `_parse_labeled` 手写解析（`nodes.py:66-80`）替换为 `model.with_structured_output(IntentSchema)`，字段用 pydantic 约束 `intent: Literal["file_read","chat","search"]`
- [ ] **`step` / `intent` 收敛为 Literal**：`state.py:11,13` 目前是自由字符串，易写错；换成 `Literal` + `TypedDict` 字面量，IDE 能查错
- [ ] **LangSmith tracing 接入**：`load.py` 读 `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY`，让每次调用都能在 Smith UI 上回放

## ⚡ P1 · 流式与交互体验

- [ ] **token 级流式**：CLI 目前按节点粒度打印 `⏺ Node(detail)`，换成 `graph.astream(..., stream_mode=["updates","messages"])` 让 `answer` 节点的 token 边生成边打印（还能展示 tool_call 流）
- [ ] **统一 HITL 入口**：目前只有 shell 走 `interrupt()`（`nodes.py:357`），把 `file_read`（覆盖已有大文件？）和未来写文件类工具都挂到同一个 HITL 协议上
- [ ] **/resume 与 /threads**：新增斜线命令列出 checkpointer 里历史 thread，支持恢复某个中断过的会话
- [ ] **/undo**（time-travel）：基于 LangGraph 的 `get_state_history` + `update_state`，让用户在 REPL 里回退到上一步重跑

## 🧠 P1 · 图结构本身

- [ ] **`understand` 并行预取**：意图判定的同时并行触发 Tavily 轻量检索（`basic`, max_results=2）作为预热缓存，减少串行等待。用 `Send` API 扇出
- [ ] **子图拆分**：把 `answer ⇄ tools` 的循环封成一个 `tool_loop_subgraph`，主图只感知"生成答案"这一单节点，便于替换/单测
- [ ] **`sorcery` 扩到 file_read / chat**：当前仅 search 路径能改写重跑（`nodes.py:240`），给 file_read 加"读取失败时引导用户修正路径"、给 chat 加"回答过空时兜底一次搜索"
- [ ] **多轮 retry 策略**：`retry_count >= 1` 的硬阈值（`nodes.py:248`）改为指数退避 + 最多 N 次，并记录每次改写的 query，避免抖动

## 🛠️ P2 · 工具层

- [ ] **工具调用超时 & 重试**：`tools_node`（`nodes.py:322`）里 `tools_by_name[name].invoke(args)` 裸调用，给 `check_weather` / `lookup_ip` / MCP 工具加 asyncio.timeout 与一次重试
- [ ] **工具结果缓存**：同一 thread 内对同参数的 `check_weather` / `lookup_ip` 命中缓存（LRU，TTL 60s），减少重复外部调用
- [ ] **`search_results` 结构化**：目前拼成一坨字符串（`nodes.py:156-173`），改为 `list[SearchItem]`，渲染与评估都更稳
- [ ] **MCP 预热 & 健康检查**：`_mcp_manager().list_tools()` 在每次 `generate_answer_node` 都调（`nodes.py:202`），挪到启动期缓存 + 定时健康探测

## 🧪 P2 · 测试与质量

- [ ] **单元测试骨架**：`tests/` 下加 pytest，mock `model` / `tavily_client`，对每个节点独立测 `state in → state out`
- [ ] **图级快照测试**：用 `graph.get_graph().draw_mermaid()` 生成的图做 golden test，防止误改边
- [ ] **类型检查**：pyproject 里加 `mypy` + `types-*`，state 和 tools 签名补齐
- [ ] **GitHub Actions CI**：lint（ruff）+ 测试 + 构建 web 静态页发 Pages

## 📚 P3 · 文档与示例

- [ ] **Mermaid 流程图**：把 README / web 里的 ASCII 流程图换成 Mermaid，用 `app.get_graph().draw_mermaid()` 自动生成
- [ ] **cookbook/**：加几个可跑例子（纯 chat / 读 CSV / 搜索 + 工具链 / MCP 接入 stdio server）
- [ ] **贡献指南 CONTRIBUTING.md**：如何加节点、加工具、加 MCP server
