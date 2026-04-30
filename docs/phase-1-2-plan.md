# Phase 1–2 文件级 diff 计划（按 LangGraph 文档修订）

> 修订点（相对前一版）：
> - Phase 1 的 react 子图直接复用 `SearchState`（共享状态键），可作为 compiled subgraph 直接 `add_node`，无需 wrapper 函数。
> - 子图不带自己的 checkpointer（per-invocation 模式），自动继承父图 checkpointer 以支持 `interrupt()`。
> - Phase 2 的 SQL 工具用 `ToolRuntime[ContextSchema]` 取 runtime context（`runtime` 是保留参数，LLM schema 中自动隐藏），不再用 `InjectedToolArg`。
> - 非 HITL 工具改用 `ToolNode` 执行（它自动注入 `runtime`）；HITL 工具仍走自定义流程。

---

## Phase 1 · 抽出 `react_subgraph`（行为不变）

### 🆕 新增 `askanswer/react.py`

```python
from langgraph.graph import END, START, StateGraph
from .schema import ContextSchema
from .state import SearchState
# ↓ 从 nodes.py 搬过来的内部函数（保持私有命名）
from ._react_internals import (
    _answer_node, _shell_plan_node, _tools_node, _route_from_answer,
)

def build_react_subgraph():
    builder = StateGraph(SearchState, context_schema=ContextSchema)
    builder.add_node("answer", _answer_node)
    builder.add_node("shell_plan", _shell_plan_node)
    builder.add_node("tools", _tools_node)

    builder.add_edge(START, "answer")
    builder.add_conditional_edges(
        "answer",
        _route_from_answer,            # "shell_plan" | "tools" | END
        {"shell_plan": "shell_plan", "tools": "tools", END: END},
    )
    builder.add_edge("shell_plan", "tools")
    builder.add_edge("tools", "answer")
    return builder.compile()            # ← 不传 checkpointer，per-invocation 模式
```

要点：
- **`StateGraph(SearchState, ...)`** —— 与父图同 schema，共享 `messages` / `pending_shell` / `final_answer` / `step` / `intent` / `user_query` / `search_results` 等字段。
- **不调用 `.compile(checkpointer=...)`** —— 子图自动继承父图的 `InMemorySaver`，`interrupt()` 一次中断、`Command(resume=...)` 直接送回，行为与现状一致（LangGraph docs 「Subgraph persistence · Per-invocation」）。
- `_route_from_answer` 把现 `route_from_answer` 中的 `"sorcery"` 出口改成 `END`，让父图接管后续路由。

### ✂️ `askanswer/nodes.py` 拆分

把以下符号搬到新文件 `askanswer/_react_internals.py`（保持 react.py 整洁）：
- `SHELL_TOOL_NAME`
- `_mcp_tool_specs` / `_mcp_tool_names`
- `generate_answer_node` → `_answer_node`
- `shell_plan_node` → `_shell_plan_node`
- `tools_node` → `_tools_node`
- `_run_shell_with_confirmation` / `_parse_decision` / `_truthy`
- `route_from_answer` → `_route_from_answer`（改 `"sorcery"` → `END`）

`nodes.py` 保留：`understand_query_node`、`file_read_node`、`sql_agent_node`、`tavily_search_node`、`sorcery_answer_node`、所有 intent 分类辅助。

### ✂️ `askanswer/graph.py` 改动

```diff
-from .nodes import (
-    SHELL_TOOL_NAME,
-    file_read_node,
-    generate_answer_node,
-    shell_plan_node,
-    sorcery_answer_node,
-    sql_agent_node,
-    tavily_search_node,
-    tools_node,
-    understand_query_node,
-)
+from .nodes import (
+    file_read_node,
+    sorcery_answer_node,
+    sql_agent_node,
+    tavily_search_node,
+    understand_query_node,
+)
+from .react import build_react_subgraph
```

```diff
-def route_from_answer(state: SearchState):
-    if state["step"] != "tool_called":
-        return "sorcery"
-    tcs = getattr(state["messages"][-1], "tool_calls", None) or []
-    if any(tc.get("name") == SHELL_TOOL_NAME for tc in tcs):
-        return "shell_plan"
-    return "tools"
```

```diff
 def create_search_assistant():
     workflow = StateGraph(SearchState, context_schema=ContextSchema)
+    react = build_react_subgraph()           # 共享 SearchState 的 compiled subgraph

     workflow.add_node("understand", understand_query_node)
     workflow.add_node("search", tavily_search_node)
-    workflow.add_node("answer", generate_answer_node)
+    workflow.add_node("answer", react)       # 直接把 compiled subgraph 当节点
     workflow.add_node("sorcery", sorcery_answer_node)
-    workflow.add_node("shell_plan", shell_plan_node)
-    workflow.add_node("tools", tools_node)
     workflow.add_node("file_read", file_read_node)
     workflow.add_node("sql", sql_agent_node)

     workflow.add_edge(START, "understand")
     workflow.add_conditional_edges("understand", route_from_understand, {...})
     workflow.add_edge("search", "answer")
     workflow.add_edge("sql", END)
-    workflow.add_conditional_edges("answer", route_from_answer, {...})
-    workflow.add_edge("shell_plan", "tools")
-    workflow.add_edge("tools", "answer")
+    workflow.add_edge("answer", "sorcery")
     workflow.add_edge("file_read", "sorcery")
     workflow.add_conditional_edges("sorcery", route_from_sorcery, {...})

     memory = InMemorySaver()
     app = workflow.compile(checkpointer=memory)
     return app
```

### ✂️ `askanswer/state.py`

无改动。

### ✂️ `askanswer/cli.py`

无改动。`stream_query` 当前用 `app.stream(..., stream_mode="updates")`：
- 子图内部的更新默认聚合在父节点 `"answer"` 名下，CLI 现有 `_render_node_update` 仍能命中 `node == "answer"` 分支。
- 如果想看到子图内每一步进度，可加 `subgraphs=True`（但事件 ns 会变成 `("answer:<uuid>",)`，需要适配 —— **Phase 1 暂不做，保持兼容**）。

### ✅ Phase 1 验收

- 命令行行为完全不变：`askanswer "你好"` / `"搜一下今天上海天气"` / `"!ls"` / shell HITL `y`/`n`/`e`。
- `askanswer --graph` 生成的 mermaid 中 `answer` 是双线框（compiled subgraph 标识）。
- 单测：mock model + mock tool，直接 `react = build_react_subgraph(); react.invoke({"messages": [...]})`，验证 tool_call → tools → answer → END 闭环 + `interrupt()` 中断恢复。
- `gh pr diff` 期望影响：删除 `nodes.py` ~200 行，新增 `react.py` ~10 行 + `_react_internals.py` ~200 行（搬迁），`graph.py` 净减少 ~10 行。

### Phase 1 风险

- **stream_mode 兼容性**：`app.stream(stream_mode="updates")` 默认不带 `subgraphs=True`，子图事件以聚合形式上报为 `"answer"` 节点更新；现 CLI 的 `_render_node_update` 依赖 `update.get("step")` / `update.get("final_answer")` 这些字段都在共享 schema 上，应该平滑。但 `update.get("intent")` 在子图里不会改写，CLI 里那条分支不受影响。回归测试要看每个 node 标记是否还出现。
- **`interrupt()` 透传**：LangGraph 文档明确支持子图 `interrupt()` 透传到父图 stream（per-invocation 模式继承父 checkpointer）。CLI 现有 `_extract_interrupt_value` / `_pending_interrupt` 兜底逻辑无需改动，但要写一个回归测试覆盖。

---

## Phase 2 · 引入 ToolRegistry + SQL 工具化

### 🆕 新增 `askanswer/registry.py`

```python
from dataclasses import dataclass
from threading import Lock
from langchain_core.tools import BaseTool, StructuredTool
from .mcp import get_manager as _mcp_manager

@dataclass(frozen=True)
class ToolDescriptor:
    tool: BaseTool
    bundles: frozenset[str]                  # {"chat","search","file","sql","mcp"} 的子集
    source: str                              # "builtin" | "sql" | "shell" | "mcp:<server>"
    requires_confirmation: bool = False

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDescriptor] = {}
        self._lock = Lock()

    def register(self, descriptor: ToolDescriptor) -> None: ...
    def unregister_source_prefix(self, prefix: str) -> None: ...   # e.g. "mcp:"
    def list(self, bundle: str | None = None) -> list[BaseTool]: ...
    def get(self, name: str) -> ToolDescriptor | None: ...
    def confirmation_names(self) -> set[str]: ...
    def refresh_mcp(self) -> None:
        with self._lock:
            for name, desc in list(self._tools.items()):
                if desc.source.startswith("mcp:"):
                    del self._tools[name]
        for spec in _mcp_manager().list_tools():
            tool = _wrap_mcp_tool(spec)
            self.register(ToolDescriptor(
                tool=tool,
                bundles=frozenset({"chat","search","file","sql","mcp"}),
                source=f"mcp:{spec['server']}",
            ))

_registry: ToolRegistry | None = None
_registry_lock = Lock()

def get_registry() -> ToolRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ToolRegistry()
            _seed_builtin(_registry)
            _seed_sql(_registry)
        return _registry
```

**`_wrap_mcp_tool`**：用 `StructuredTool.from_function` + `args_schema` 由 jsonschema 转 pydantic（best-effort：失败时退回无 schema 的 `Tool`）。这样 MCP 工具进 `bind_tools` 与 built-in 完全等价，删除 `_mcp_tool_specs` 这条分支。

### 🆕 新增 `askanswer/sqlagent/sql_tool.py`

```python
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, ToolRuntime           # ← 关键：ToolRuntime
from ..schema import ContextSchema, normalize_context
from .sql_agent import extract_sql_answer, run_sql_agent

@tool
def sql_query(question: str, runtime: ToolRuntime[ContextSchema]) -> str:
    """查询数据库回答自然语言问题。需要 runtime context 中提供 db_dsn。"""
    context = normalize_context(runtime.context)
    messages = run_sql_agent([HumanMessage(content=question)], context=context)
    return extract_sql_answer(messages)
```

要点：
- `runtime: ToolRuntime[ContextSchema]` 是 LangGraph 保留参数，**模型看到的 schema 中只会有 `question`**（docs 「Tools · Reserved argument names」）。
- 必须由 `ToolNode` 或在 LangGraph 执行上下文中调用，否则 `runtime` 不会被注入。

### ✂️ `askanswer/_react_internals.py` 改动

**`_answer_node`** —— 按 intent 选 bundle，丢掉 dict-spec 分支：

```diff
-from .mcp import get_manager as _mcp_manager
-from .tools import tools, tools_by_name
+from .registry import get_registry
```
```diff
-    mcp_specs = _mcp_tool_specs()
-    mcp_line = ""
-    if mcp_specs:
-        names = ", ".join(spec["function"]["name"] for spec in mcp_specs)
-        mcp_line = f"\n额外的 MCP 工具可直接按名称调用：{names}。"
-    ...
-    bound = model.bind_tools(tools + mcp_specs) if mcp_specs else model.bind_tools(tools)
+    bundle = state.get("intent") or "chat"
+    bundle_tools = get_registry().list(bundle=bundle)
+    bound = model.bind_tools(bundle_tools)
```

`_mcp_tool_specs` / `_mcp_tool_names` 整体删除。

**`_tools_node`** —— 拆成「ToolNode 跑普通工具」+「自定义跑 HITL 工具」：

```python
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from .schema import ContextSchema

def _tools_node(state: SearchState, runtime: Runtime[ContextSchema]) -> dict:
    registry = get_registry()
    confirmation_names = registry.confirmation_names()
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []

    confirm_calls = [tc for tc in tool_calls if tc["name"] in confirmation_names]
    plain_calls   = [tc for tc in tool_calls if tc["name"] not in confirmation_names]

    out_messages: list = []

    if plain_calls:
        # 让 ToolNode 处理 runtime / state 注入、并发执行、错误兜底
        plain_tools = [registry.get(tc["name"]).tool for tc in plain_calls]
        node = ToolNode(plain_tools)
        # 用一个仅含 plain_calls 的 AIMessage 喂给 ToolNode
        sub_msg = last_msg.model_copy(update={"tool_calls": plain_calls})
        result = node.invoke(
            {"messages": state["messages"][:-1] + [sub_msg]},
            config={"configurable": {}},  # 父图 config 自动传播
        )
        out_messages.extend(result["messages"][len(state["messages"][:-1] + [sub_msg]):])

    for tc in confirm_calls:
        out_messages.append(_run_with_confirmation(tc, state))   # 仍是 ToolMessage

    return {"messages": out_messages, "pending_shell": {}}
```

> **细节**：`ToolNode` 自动给每个 tool_call 调用对应工具并返回 `ToolMessage`，并把 `runtime` 注入到带 `ToolRuntime` 参数的工具里 —— 这正是 `sql_query` 拿到 `runtime.context` 的关键。
> 上面 `model_copy` 的小操作是为了避免 ToolNode 对未在它工具集里的 `confirm_calls` 报「unknown tool」。

**`_run_shell_with_confirmation` → `_run_with_confirmation`**：函数体不变，把硬编码的 `SHELL_TOOL_NAME` 解耦成「从 descriptor 拿 confirmation planner」。短期 shell 仍是唯一例子，但 API 已是通用的：

```python
def _run_with_confirmation(tool_call: dict, state: SearchState) -> ToolMessage:
    descriptor = get_registry().get(tool_call["name"])
    planner = getattr(descriptor, "confirmation_planner", None) or _shell_command_planner
    plan = (state.get("pending_shell") or {}).get(tool_call["id"]) or planner(tool_call)
    ...   # 余下与现 _run_shell_with_confirmation 等价
```

**`_shell_plan_node`**：路由触发条件改为「tool_call 命中 `confirmation_names`」（不再硬编码 shell 名）：

```diff
-from .tools import gen_shell_command_spec
-...
-for tc in state["messages"][-1].tool_calls:
-    if tc["name"] != SHELL_TOOL_NAME: continue
+confirmation_names = get_registry().confirmation_names()
+for tc in state["messages"][-1].tool_calls:
+    if tc["name"] not in confirmation_names: continue
```

**`_route_from_answer`** 同步：

```diff
-tcs = getattr(state["messages"][-1], "tool_calls", None) or []
-if any(tc.get("name") == SHELL_TOOL_NAME for tc in tcs):
-    return "shell_plan"
+confirmation_names = get_registry().confirmation_names()
+tcs = getattr(state["messages"][-1], "tool_calls", None) or []
+if any(tc.get("name") in confirmation_names for tc in tcs):
+    return "shell_plan"
 return "tools"
```

### ✂️ `askanswer/graph.py` —— SQL 收敛

```diff
 def route_from_understand(state):
     intent = state.get("intent", "search")
     if intent == "file_read": return "file_read"
-    if intent == "sql":       return "sql"
-    if intent == "chat":      return "answer"
+    if intent in ("chat", "sql"): return "answer"   # sql 走 react + sql_query 工具
     return "search"
```
```diff
-    workflow.add_node("sql", sql_agent_node)
-    workflow.add_edge("sql", END)
```

`add_conditional_edges` 的 mapping 同步去掉 `"sql"`。

### ✂️ `askanswer/nodes.py`

- 删除 `sql_agent_node` 与对 `sql_agent` 的 import。
- intent 分类逻辑保留 —— `intent="sql"` 仍然有用，用于在 `_answer_node` 里选 `bundle="sql"`，让模型只看到 `sql_query` 这一个工具。

### ✂️ `askanswer/cli.py`

```diff
 from .mcp import get_manager as _mcp_manager, shutdown_manager as _mcp_shutdown
+from .registry import get_registry
```

`/mcp` 增删后刷新缓存：

```diff
 def _add_mcp_url(url, name):
     ...
     registered = _mcp_manager().add_url(url, name=name)
+    get_registry().refresh_mcp()
     ...

 def _remove_mcp_server(name):
     ok = _mcp_manager().remove(name)
+    if ok:
+        get_registry().refresh_mcp()
```

`main()` 启动时预热（避免首问时重 IO）：

```diff
 atexit.register(_mcp_shutdown)
+get_registry()                         # built-in + sql 注册
```

### ✂️ `askanswer/tools.py`

无结构性改动。registry 注册时给 `gen_shell_commands_run` 打 `requires_confirmation=True`。

### ✅ Phase 2 验收

- **行为面**：
  - 6 个 built-in 工具 + shell 仍可在 chat/search 流程被调用，HITL 弹确认。
  - "查一下用户表的数量"：意图 `sql` → react 子图（仅绑 `sql_query`）→ 模型调用 `sql_query` → `ToolNode` 自动注入 `runtime` → 内部 `run_sql_agent` 拿到 `ContextSchema(db_dsn=...)`。
  - `/mcp <url>` → `registry.refresh_mcp()` → 立即可用。
- **代码面**：
  - 生产代码不再引用 `_mcp_tool_specs` / `_mcp_tool_names` / `tools_by_name` / `SHELL_TOOL_NAME`（仅 `tools.py` 自身保留这些工具的源头）。
  - `app.get_graph().draw_mermaid()` 中已无 `sql` 节点。

### Phase 2 风险与应对

- **`langchain.tools.ToolRuntime` / `ToolNode` 注入版本要求**：`runtime` 注入需 `langgraph >= 1.1.5` 或 `deepagents >= 0.5.0`（docs 「Execution info / Server info」备注）。pyproject 里 `langgraph` 没钉版本，**Phase 2 第一步是把 `langgraph>=1.1.5` 加进 dependencies**。
- **MCP `args_schema` 转换**：jsonschema → pydantic 边角情况（嵌套 `anyOf`、`$ref`）易炸；`_wrap_mcp_tool` 里包 `try/except`，失败时退化为参数全 `Any` 的简单 `Tool`，并 warn 一次。
- **`ToolNode` 与并发 `tool_calls`**：`ToolNode` 默认并行执行 plain_calls，`pending_shell` 已经只关心 confirm_calls，不会冲突。但 `messages` reducer 是 `add_messages`，确认顺序问题可由 LangGraph 自身处理。
- **`stream_mode="updates"` 视图**：`ToolNode` 每个工具会发独立 update，CLI 现在打的 `⏺ Tools` 标记会更频繁。可在 `_render_node_update` 里聚合，或加 `subgraphs=False`（保持单条） —— Phase 2 默认保持单条聚合。
- **行为差异**：原 SQL 路径直达 `END`，现在多走一次 LLM（react `_answer_node` 决定调 `sql_query`）。延迟 +1 LLM call，但获得了「读完 SQL 结果再追问 / 再调其它工具」的能力。后续 Phase 4 sorcery 可加快路径。

---

## 落地顺序建议

1. **Phase 1 单 PR**：纯重构，3 文件改动 + 2 新文件。回归 CLI 行为不变即可合入。
2. **Phase 2 拆两 PR**：
   - **2a**：`registry.py` + `_wrap_mcp_tool` + 把 built-in/MCP 接进去 + `_answer_node` 改读 registry（SQL 仍走老路）。
   - **2b**：`sql_tool.py` + `_tools_node` 切到 `ToolNode` + 删除 `sql_agent_node` / `sql` 路由 / `SHELL_TOOL_NAME` 硬编码。

每一步都能独立通过 `askanswer --graph` mermaid + 一组手工提问回归。
