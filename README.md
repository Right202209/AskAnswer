# AskAnswer

基于 LangGraph 的命令行智能助手：按意图分流到「读本地文件 / 联网搜索 / 直接回答」，由 LLM 调度工具并整合结果，支持自评改写与 Markdown 渲染。

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
├── cli.py        # 命令行入口：欢迎框、斜线命令、流式进度、Markdown 渲染
├── graph.py      # LangGraph 工作流编排
├── nodes.py      # 节点实现：理解分流 / 搜索 / 读文件 / 回答 / 自评 / 工具
├── state.py      # SearchState 定义（含 intent / file_path）
├── tools.py      # 工具集
├── mcp.py        # MCP 客户端管理器：URL (HTTP/SSE) + stdio，多服务聚合
├── load.py       # 模型、Tavily、API key 加载
└── __main__.py   # python -m 入口
```

## 工作流

```
START → understand ─┬─ file_read ───────────┐
                    ├─ sql ──────────────── END
                    ├─ answer (chat) ───────┤
                    └─ search → answer ─────┤
                         answer ⇄ tools     │
                                            ▼
                                         sorcery ─ {search 重搜 | END}
```

- `understand`：分类用户意图为 `file_read` / `sql` / `chat` / `search`，并提取文件路径或搜索关键词
- `file_read`：直接调用 `read_file` 工具读取并分析本地文件
- `sql`：从 runtime context 读取数据库配置，调用 SQL agent 查询数据库
- `search`：Tavily 检索 Top 5
- `answer`：模型已绑定工具，可按需调用；整合搜索 / 工具结果给出回答
- `tools`：执行模型发起的工具调用，结果回写后重入 `answer`
- `sorcery`：自评答案质量；仅 `search` 路径允许改写关键词重跑一次

## 工具集

| 工具 | 说明 |
| --- | --- |
| `read_file` | 读取 `.txt` / `.md` / `.json` / `.csv` / `.xlsx` 并由 LLM 分析 |
| `check_weather` | OpenWeather 实时天气 |
| `get_current_time` | 指定时区当前时间 |
| `calculate` | 安全表达式计算（+ - * / // % ** 与括号） |
| `convert_currency` | 货币汇率换算 |
| `lookup_ip` | IP 地理位置与运营商查询 |

模型在 `answer` 节点已 `bind_tools(tools)`，会按需自行调用；`tools_node` 负责执行并回传 `ToolMessage`。
