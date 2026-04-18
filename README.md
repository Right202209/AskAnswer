# AskAnswer

基于 LangGraph 的命令行搜索助手：理解问题 → Tavily 搜索 → LLM 整合 → 自评改写。

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
askanswer                         # 交互模式（exit/quit/q 退出）
python -m askanswer               # 等价入口
```

## 结构

```
askanswer/
├── cli.py        # 命令行入口与交互循环
├── graph.py      # LangGraph 工作流编排
├── nodes.py      # 各节点：理解 / 搜索 / 生成 / 自评 / 工具
├── state.py      # SearchState 定义
├── tools.py      # 工具集（天气 / 时间 / 计算 / 汇率 / IP）
├── load.py       # 模型、Tavily、API key 加载
└── __main__.py   # python -m 入口
```

## 工作流

```
START → understand → search → answer → sorcery → (END | search)
                                  ↘ tools ↗
```

- `understand`：提炼搜索词
- `search`：Tavily 检索 Top 5
- `answer`：基于结果生成回答（失败则回退到 LLM 知识）
- `sorcery`：自评，必要时改写搜索词重跑一次（最多一次重试）
- `tools`：执行模型发起的工具调用
