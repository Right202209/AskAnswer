# AskAnswer 模块级易出 Bug 点审查

日期：2026-05-08  
范围：根项目 `askanswer` 包与 `web` 静态页。`ouroboros/` 是独立子项目（有单独 `pyproject.toml`），本报告未展开。

视角：假设攻击者或恶意用户想用输入、模型输出、环境变量、外部服务和异常状态把系统搞坏。每个模块列 3 个最可能出问题的点。

## 1. CLI / REPL 模块

1. `!<cmd>` 直通 shell 的危险检查是黑名单，容易漏掉等价破坏命令。`run_bang_command()` 在确认后用 `execute_shell_command(..., shell=True)` 执行完整 shell 语法；黑名单虽覆盖 `rm/rmdir/shutdown/reboot/halt/poweroff/mkfs/dd/sudo/fork-bomb/重定向覆盖/kill -9/chmod -R/chown -R/mv 到根/写磁盘设备` 一长串模式，但绕过面仍大，例如破坏性工具不叫这些名字、通过解释器/脚本执行、或借助 shell 特性组合。见 `askanswer/cli.py:664`、`askanswer/cli.py:692`、`askanswer/tools.py:239`。
2. `/mcp <url>` 会把任意用户输入的 URL 接成外部工具源，连接成功后刷新注册表。恶意 MCP server 可以发布诱导性工具描述，并被后续 LLM 在所有 intent 中调用。见 `askanswer/cli.py:839`、`askanswer/cli.py:848`、`askanswer/registry.py:41`。
3. 恢复含挂起 shell interrupt 的会话时只提示不阻断。攻击者可以制造挂起状态，再诱导用户切线程继续对话，导致旧确认、当前输入和 LangGraph checkpoint 交错，出现执行对象错位或状态难以预测。见 `askanswer/cli.py:1022`、`askanswer/cli.py:1043`。

## 2. 内置工具模块

1. `read_file()` 接受任意路径并启用 `MarkItDown(enable_plugins=True)`，没有路径白名单、大小限制、扩展限制或敏感文件拦截。恶意提示可以诱导读取密钥、历史数据库、SSH 配置，或喂入超大/畸形文件拖垮进程。见 `askanswer/tools.py:166`、`askanswer/tools.py:183`。
2. `execute_shell_command()` 有 30 秒整体超时（`tools.py:336`），但没有工作目录隔离、环境裁剪或输出上限。即使 `shell=False`，30 秒内一条高吞吐命令也能把内存和终端打爆；`shell=True` 路径还直接保留管道、替换、控制流等语法。见 `askanswer/tools.py:315`。
3. `calculate()` 允许指数运算但没有复杂度上限。类似超大幂、深层表达式或大整数运算可能造成 CPU/内存 DoS。见 `askanswer/tools.py:87`、`askanswer/tools.py:101`。

## 3. 工具注册表模块

1. 内置普通工具默认暴露给所有 intent，`read_file`、联网搜索、IP 查询等都能在 chat/sql/math 场景被模型调用。只要 prompt 注入让模型偏航，就可能越权触发不相关工具。见 `askanswer/registry.py:36`、`askanswer/registry.py:189`。
2. MCP 工具统一对所有 intent 开放，且没有人工确认等级。外部工具可能执行付费 API、写操作或泄露信息，但注册时全部按普通工具处理。见 `askanswer/registry.py:41`、`askanswer/registry.py:138`。
3. MCP schema 转 pydantic 的逻辑只支持浅层 schema，复杂 schema 直接跳过。攻击者或不规范 server 可以让关键安全参数无法表达，或者让工具悄悄不可用，造成行为与 UI 展示不一致。见 `askanswer/registry.py:265`、`askanswer/registry.py:302`。

## 4. MCP 客户端模块

1. `add_stdio()` 能启动任意本地命令作为 MCP server。虽然 CLI 当前只暴露 URL，但公开 API 没有命令白名单；一旦被插件或外部调用接上，就是本地代码执行边界。见 `askanswer/mcp.py:102`。
2. `_submit()` 阻塞等待 future，没有超时。恶意或卡死的 MCP server 可以让主线程永久挂住，影响 `/mcp list`、工具调用和退出清理。见 `askanswer/mcp.py:455`。
3. HTTP/SSE MCP 连接没有 URL allowlist、TLS 策略、认证提示或工具调用超时。攻击者控制的 endpoint 可以长期占用连接、返回超大工具结果，或利用工具描述做 prompt 注入。见 `askanswer/mcp.py:308`、`askanswer/mcp.py:196`。

## 5. 图编排 / ReAct 模块

1. `_answer_node()` 把所有工具名直接放入 system prompt，但没有把工具风险分级传给模型。模型只看到“可用工具”，不知道哪些工具应少用或需要用户授权，容易被用户 prompt 注入引向高风险工具。见 `askanswer/_react_internals.py:55`。
2. 普通工具在 `_tools_node()` 中由 `ToolNode` 并发执行，除 shell 外没有确认分支。MCP、读文件、外部 API 都属于普通工具，异常虽然被包装，但副作用已经发生。见 `askanswer/_react_internals.py:155`。
3. shell HITL 只在执行前用同一套黑名单复查命令。用户在确认界面重新生成/修改命令后仍可能通过未覆盖的危险模式，且执行结果会回填给模型继续推理。见 `askanswer/_react_internals.py:251`、`askanswer/_react_internals.py:299`。

## 6. 父图节点 / 自评模块

1. `understand_query_node()` 直接取 `state["messages"][-1].content`，假设最后一条总是用户消息且 content 是字符串。异常状态、恢复中的 ToolMessage 或多模态 content 都可能让分类崩。见 `askanswer/nodes.py:58`。
2. LLM 分类失败时 fallback 可能把长文本误判为 search（`len(text) > 80` 且不含问号即归 search）。攻击者可以用长提示把本应本地回答的内容推向联网工具，扩大外部依赖和 prompt 注入面。见 `askanswer/intents/__init__.py:84`。
3. `sorcery_answer_node()` 完全信任 handler.evaluate 的结构。搜索自评由 LLM 判定是否重搜，攻击者可以通过搜索结果文本诱导重试查询，造成额外成本和偏航。见 `askanswer/nodes.py:99`、`askanswer/intents/search.py:68`。

## 7. Intent 分类模块

1. 文件路径正则会从自然语言里抓取路径，但不校验路径是否位于工作区，也不处理空格路径、glob、符号链接和同名敏感文件。最终会把风险传给 `read_file`。见 `askanswer/intents/base.py:69`、`askanswer/intents/base.py:84`。
2. SQL intent 会把包含 `insert/update/delete` 的用户输入识别为 SQL 请求。后续虽然 prompt 说不要 DML，但没有硬拦截，攻击者正好可以把破坏性 SQL 带入 SQL agent。见 `askanswer/intents/sql.py:24`、`askanswer/intents/sql.py:36`。
3. Chat intent 的前缀词很宽，例如“如何/怎么/为什么”。很多需要联网核验或涉及工具安全的问题会被归为 chat，再由 chat bundle 暴露的通用工具继续决定是否调用，分类边界不稳定。见 `askanswer/intents/chat.py:9`、`askanswer/intents/chat.py:37`。

## 8. SQL Agent 模块

1. 禁止 DML 只写在 prompt 里，没有 SQL AST/只读事务/数据库权限层拦截。模型或 prompt 注入一旦生成 `DROP/UPDATE/DELETE`，`sql_db_query` 仍可能执行。见 `askanswer/sqlagent/sql_node.py:186`、`askanswer/sqlagent/sql_node.py:143`。
2. 数据库连接按 DSN 直接缓存，来自 runtime context 或环境变量。恶意/错误 DSN 可以连到不该访问的数据库；缓存还可能让凭证轮换、租户隔离和连接生命周期变复杂。见 `askanswer/sqlagent/sql_interact.py:20`、`askanswer/sqlagent/sql_interact.py:30`。
3. 表结构和查询结果只做字符截断，不做列级权限、敏感字段脱敏或行级隔离。攻击者只要能让 agent 查 schema 或宽泛查询，就可能把隐私列塞回 LLM 上下文。见 `askanswer/sqlagent/sql_node.py:18`、`askanswer/sqlagent/sql_node.py:103`。

## 9. Helix 规格演化模块

1. 非交互环境下访谈会自动采用 LLM 生成的 `default_answer`，没有用户确认。攻击者可以通过模糊需求影响默认假设，让系统自说自话地产出错误规格。见 `askanswer/helix/nodes.py:118`、`askanswer/helix/nodes.py:151`。
2. seed、execute、evaluate 都由同一个模型链路生成并自评，缺少独立校验。恶意 topic 可以让 artifact 与 evaluation 同时偏航，出现“自评通过但规格错误”。见 `askanswer/helix/nodes.py:186`、`askanswer/helix/nodes.py:226`、`askanswer/helix/nodes.py:241`。
3. 模糊需求识别只靠启发式。稍长文本、带 URL/路径/算式的需求会绕过 Helix；短而具体的实现请求也可能被误判为 Helix，导致用户想直接执行时被拖入访谈循环。见 `askanswer/intents/helix.py:84`、`askanswer/intents/helix.py:94`。

## 10. 持久化模块

1. `ASKANSWER_DB_PATH` 可把 SQLite 状态库指向任意可写路径，没有权限检查或路径约束。恶意环境变量能把历史写到共享目录，或覆盖/污染用户指定文件。见 `askanswer/persistence.py:34`。
2. `delete_thread()` 拼接内部表名执行删除，当前表名是常量所以没有 SQL 注入，但强依赖 LangGraph SQLite 表结构；升级后表名或列变化可能静默跳过，造成“显示已删除但 checkpoint 仍在”。见 `askanswer/persistence.py:238`。
3. `_row_to_meta()` 只对 `tags`（JSON → list 校验）和时间戳（`int(... or 0)`）做了基础类型矫正；`title/preview/last_intent/model_label` 直接原样回填、没有长度或类型校验。被污染的本地状态库可以塞入超大/异常类型的这些字段，让 `/threads` 渲染变慢或崩溃。见 `askanswer/persistence.py:328`。

## 11. 模型 / 配置模块

1. `load_dotenv(override=True)` 让项目 `.env` 覆盖已有环境变量。进入不可信目录运行时，仓库内 `.env` 可以劫持 API key、DSN、模型配置等运行时行为。见 `askanswer/load.py:13`。
2. `/model` 接受任意 provider/spec 并直接调用 `init_chat_model`。错误或恶意 provider 可能导致运行时异常、走错供应商，或触发未预期的网络调用。见 `askanswer/load.py:83`、`askanswer/cli.py:803`。
3. `ContextSchema` 包含 `tenant_id`，但 SQL、工具和持久化层基本没有强制使用它。调用方以为已传租户，实际查询和工具执行仍可能跨租户。见 `askanswer/schema.py:16`、`askanswer/schema.py:19`。

## 12. 状态模型模块

1. `SearchState` 是 `TypedDict`，运行时没有校验。任意节点返回错误类型都能进入下游，例如 `pending_shell` 不是 dict、`messages` 混入非 Message 对象。见 `askanswer/state.py:7`。
2. `messages` 使用 `add_messages` 只追加不覆盖，长期会话会持续膨胀。攻击者可以用大工具输出或重复会话拖大 checkpoint 和 prompt，造成成本与延迟问题。见 `askanswer/state.py:10`。
3. `pending_shell` 跨 interrupt 持久化，但只在 `_tools_node()` 正常返回后清空。异常、恢复或切线程时可能留下旧命令计划，后续确认界面读取到过期命令。见 `askanswer/state.py:20`、`askanswer/_react_internals.py:233`。

## 13. UI 输入 / 选择模块

1. prompt 历史默认写到 `~/.askanswer/history` 或 `ASKANSWER_DB_PATH` 同目录。用户输入可能包含密钥、数据库 DSN 或敏感 SQL，被明文保存。见 `askanswer/ui_input.py:77`、`askanswer/ui_input.py:130`。
2. 非 TTY fallback 的 `_numbered_select()` 默认回车选择默认项。若调用方默认值设错，管道/CI 环境可能在无人确认时走到“执行”或不安全选项；当前 shell 确认默认取消，但这是调用方约定，不是组件保证。见 `askanswer/ui_select.py:160`。
3. `_arrow_select()` 直接操作终端 cbreak 和 ANSI 光标保存/恢复，异常路径虽恢复 termios，但如果输出被其他线程污染，菜单可能覆盖重要确认文本，导致误选。见 `askanswer/ui_select.py:84`。

## 14. 静态 Web 模块

1. `web/index.html` 的功能描述落后于 README：只写“读文件 / 联网搜索 / 直接回答”，没提 SQL、Helix、持久化等。用户按网页理解安全边界会低估实际能力。见 `web/index.html:14`。
2. 页面宣传 MCP 热接入和 Shell 前缀，但没有任何安全提示。攻击者诱导用户访问文档并按说明连接 MCP 或执行 shell 时，缺少风险提醒。见 `web/index.html:35`、`web/index.html:47`。
3. workflow 图仍是旧拓扑，显示独立 `file_read/search` 节点，但代码已合并到 ReAct 工具流。文档偏差会让调试和威胁建模都看错路径。见 `web/index.html:79`、`web/index.html:81`。

## 建议优先修复顺序

1. 工具安全边界：给 `read_file` 加路径/大小限制，给 shell 和 MCP 工具做 allowlist、超时、输出上限和确认分级。
2. SQL 只读保证：用数据库只读账号、只读事务或 SQL parser 硬拦截 DML/DDL，而不是只靠 prompt。
3. MCP 隔离：默认不要把 MCP 暴露给所有 intent；外部工具按风险等级要求用户确认。
4. 持久化与历史：敏感输入不写 history，或提供关闭历史/加密/脱敏选项。
5. 文档同步：更新 `web/index.html` 和 README 的工作流、安全提示、实际工具范围。
