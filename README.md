# AskAnswer

基于 LangGraph 的命令行智能助手：按意图分流到「读本地文件 / 联网搜索 / SQL 查询 / 规格演化 / 直接回答」，由 LLM 调度工具并整合结果，支持自评改写与 Markdown 渲染。

## 安装

```bash
pip install -e .
```

## 配置

在项目根目录创建 `.env`：

```
OPENAI_API_KEY=...
TAVILY_API_KEY=...
OPENWEATHER_API_KEY=...   # 可选，启用天气工具
WLANGGRAPH_POSTGRES_DSN=postgresql://user:password@localhost:5432/dbname  # 可选，SQL agent 默认数据库
ASKANSWER_DB_DIALECT=  # 可选；留空时从数据库连接自动推断方言
```

SQL agent 实际从 LangGraph runtime context 读取数据库配置：

```python
from askanswer.graph import create_search_assistant
from askanswer.schema import ContextSchema

app = create_search_assistant()
app.invoke(
    {"messages": [{"role": "user", "content": "统计一下订单数量"}]},
    context=ContextSchema(db_dsn="postgresql://user:password@localhost:5432/dbname"),
)
```

CLI 会把 `.env` 中的 `WLANGGRAPH_POSTGRES_DSN` / `ASKANSWER_DB_DIALECT` 注入 runtime context。

### Helix 规格演化子图

`helix_spec_loop` 工具背后是一个独立子图：`interview → seed → execute → evaluate → (seed 重跑 | finalize)`。当用户给出模糊需求（`"用苏格拉底问我..." / "spec-first design..." / "需求澄清..."` 等）时，意图分类器命中 `helix`，LLM 转而调用该工具：

- **interview**：按 scope/constraints/outputs/verification 四个歧义轨道生成关键问题并自答（标注 `assumption:`）
- **seed**：晶化为 Seed 规格（`goal / constraints / acceptance_criteria / ontology / principles`）
- **execute**：根据 Seed 产出方案文本（步骤、关键代码骨架或配置）
- **evaluate**：覆盖率自查 + 0–1 语义对齐分 + gaps 列表
- **演化**：rejected 时回到 seed 修补 gaps，最多 `MAX_GENERATIONS=3` 代

子图返回 Markdown：`## Goal / ## Constraints / ## Acceptance criteria / ## Artifact / ## Evaluation / ## Lineage`。设计文档见 `docs/helix-subgraph-plan.md`。

## 使用

```bash
askanswer "上海今天天气怎么样"   # 单次提问
askanswer                         # 交互模式
python -m askanswer               # 等价入口
askanswer --graph                 # 输出 LangGraph Mermaid 图
askanswer --graph docs/graph.mmd  # 写入 Mermaid 图文件
```

### 交互模式界面

启动后进入类 Claude Code 的 REPL，含圆角欢迎框、箱式输入提示、进度标记 `⏺ Node(detail)`，回答以 Markdown 渲染（标题 / 列表 / 代码块都能在主流 shell 正常显示，重定向到文件时自动退化为纯文本）。

内置斜线命令：

| 命令 | 说明 |
| --- | --- |
| `/help` | 显示帮助 |
| `/clear` | 清屏并开启新会话（新 thread） |
| `/status` | 查看当前会话信息（含已连接的 MCP 服务） |
| `/mcp` | 管理 MCP 服务（见下文） |
| `/exit` / `/quit` / `/q` | 退出（也可 Ctrl-D） |

另有快捷前缀 `!<cmd>`：直接在交互模式中执行一条 shell 命令（如 `!ls -la`、`!git status | head`），跳过 LLM 流程。命中高风险模式（`rm`、`sudo`、`dd if=` 等）时会二次确认。

### MCP 支持

通过 `/mcp <url>` 即可热接入一个 MCP 服务。URL 传输自动识别：路径以 `/sse` 结尾走 SSE，其余按 streamable-HTTP 处理；也支持从代码里直接 `add_stdio(...)` 启动子进程 MCP。

```text
/mcp https://example.com/mcp            # 连接（自动推导服务名）
/mcp https://example.com/mcp my-server  # 指定服务名
/mcp list                               # 列出已连接服务
/mcp tools [server]                     # 列出工具（可按 server 过滤）
/mcp remove <name>                      # 断开指定服务
```

多个 server 的工具在聚合时以 `<server>__<tool>` 形式暴露；`call_tool` 按前缀路由到对应 session。每个 server 的上下文管理器在同一 asyncio 任务内进入/退出，避免 anyio cancel-scope 跨任务的问题。

## 结构

```
askanswer/
├── cli.py               # 命令行入口：欢迎框、斜线命令、流式进度、Markdown 渲染
├── graph.py             # 父图编排：understand → answer(react) → sorcery
├── react.py             # answer 节点对应的 react 子图（answer ⇄ tools / shell_plan）
├── _react_internals.py  # react 子图节点实现（含意图重判 _reclassify_intent）
├── nodes.py             # 父图节点：意图识别、自评回写
├── state.py             # SearchState（含 intent / file_path / pending_shell）
├── schema.py            # ContextSchema：runtime context（db_dsn、tenant 等）
├── tools.py             # 内置工具（搜索、读文件、天气、计算、IP、shell …）
├── registry.py          # ToolRegistry：按 bundle 暴露工具，含 MCP 包装
├── mcp.py               # MCP 客户端管理器：URL (HTTP/SSE) + stdio，多服务聚合
├── sqlagent/            # SQL agent 子图，作为 sql_query 工具暴露
├── helix/               # 规格优先演化子图，作为 helix_spec_loop 工具暴露
├── intents/             # 意图 handler（chat/search/file_read/sql/math/helix）
├── load.py              # 模型、Tavily、API key 加载
└── __main__.py          # python -m 入口
```

## 工作流

```
START → understand → answer ⇄ tools / shell_plan → sorcery → {answer 重跑 | END}
```

- `understand`：本地分类器优先（关键词 / 正则）判断意图为 `file_read` / `sql` / `chat` / `search` / `math` / `helix`，仅在歧义时调用 LLM；提取文件路径或搜索关键词
- `answer`：react 子图的入口。按当前 `intent` 选择系统提示词与可见工具集（`registry.list(bundle=intent)`）；每轮还会用 `_reclassify_intent` 复判最新一条用户消息，使会话中途的话题切换（chat → SQL 等）实时切换工具集
- `tools`：执行模型发起的工具调用（`tavily_search` / `read_file` / `sql_query` / 各类内置工具 / MCP 工具 …），结果回写后重入 `answer`
- `shell_plan`：`gen_shell_commands_run` 走人机确认分支，预生成命令并 `interrupt()` 等待用户批准
- `sorcery`：自评答案质量；仅 `search` 路径允许改写关键词重跑一次（注入 `HumanMessage` 让模型重新调用 `tavily_search`）

## 工具集

模型在 `_answer_node` 已 `bind_tools(...)`，按当前意图过滤的工具集来自 `ToolRegistry`；内置工具默认对所有意图可见，shell 工具不进入 SQL bundle。

| 工具 | 说明 |
| --- | --- |
| `tavily_search` | Tavily 联网搜索 Top 5 + 综合答案 |
| `read_file` | `markitdown` 解析任意文件（txt/md/json/csv/xlsx/pdf/docx/代码 …）后由 LLM 总结 |
| `sql_query` | 自然语言转 SQL：调用 SQL agent 子图，从 runtime context 读取 DSN/方言/租户 |
| `helix_spec_loop` | 规格优先演化：苏格拉底澄清 → 生成 Seed → 产出方案 → 自评迭代（最多 3 代） |
| `check_weather` | OpenWeather 实时天气 |
| `get_current_time` | 指定时区当前时间 |
| `calculate` | 安全表达式计算（+ - * / // % ** 与括号） |
| `convert_currency` | 货币汇率换算 |
| `lookup_ip` | IP 地理位置与运营商查询 |
| `pwd` | 当前工作目录 |
| `gen_shell_commands_run` | 生成并执行单条 shell 命令；高风险模式直接拦截，其余命令在 `interrupt()` 处暂停等待用户确认/编辑 |

通过 `/mcp` 接入的 MCP 工具会以 `<server>__<tool>` 形式自动加入注册表，对所有意图可见。
