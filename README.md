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
```

## 使用

```bash
askanswer "上海今天天气怎么样"   # 单次提问
askanswer                         # 交互模式
python -m askanswer               # 等价入口
```

### 交互模式界面

启动后进入类 Claude Code 的 REPL，含圆角欢迎框、箱式输入提示、进度标记 `⏺ Node(detail)`，回答以 Markdown 渲染（标题 / 列表 / 代码块都能在主流 shell 正常显示，重定向到文件时自动退化为纯文本）。

内置斜线命令：

| 命令 | 说明 |
| --- | --- |
| `/help` | 显示帮助 |
| `/clear` | 清屏并开启新会话（新 thread） |
| `/status` | 查看当前会话信息 |
| `/exit` / `/quit` / `/q` | 退出（也可 Ctrl-D） |

## 结构

```
askanswer/
├── cli.py        # 命令行入口：欢迎框、斜线命令、流式进度、Markdown 渲染
├── graph.py      # LangGraph 工作流编排
├── nodes.py      # 节点实现：理解分流 / 搜索 / 读文件 / 回答 / 自评 / 工具
├── state.py      # SearchState 定义（含 intent / file_path）
├── tools.py      # 工具集
├── load.py       # 模型、Tavily、API key 加载
└── __main__.py   # python -m 入口
```

## 工作流

```
START → understand ─┬─ file_read ───────────┐
                    ├─ answer (chat) ───────┤
                    └─ search → answer ─────┤
                         answer ⇄ tools     │
                                            ▼
                                         sorcery ─ {search 重搜 | END}
```

- `understand`：分类用户意图为 `file_read` / `chat` / `search`，并提取文件路径或搜索关键词
- `file_read`：直接调用 `read_file` 工具读取并分析本地文件
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
