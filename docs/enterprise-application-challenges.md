# AskAnswer 企业级应用难点增强建议

当前项目是一个基于 LangGraph 的 CLI 智能助手，已经具备意图路由、工具调用、SQL Agent、MCP 接入、shell 人机确认等能力。要体现企业级应用难点，建议重点补充可治理、可追溯、可恢复、可观测、可扩展的能力。

## 1. 企业级会话持久化与审计

这是最适合当前项目的主线。仓库中已有 `docs/enterprise-persistence-plan.md`，可以继续沿着这个方向落地。

可实现能力：

- SQLite/Postgres checkpointer，支持跨进程恢复会话
- `/threads`、`/resume`、`/delete`、`/title`
- `/checkpoints`、`/undo`、`/fork` 时间旅行
- 工具调用审计：工具名、参数摘要、耗时、结果大小、是否失败
- token 和成本统计：按 thread、模型、工具聚合
- 会话导出：Markdown / JSON

企业级难点：

- 状态持久化
- checkpoint 一致性
- 会话恢复
- 审计追踪
- 成本治理

## 2. 多租户与权限隔离

如果要把项目从 CLI 工具升级成企业平台，可以增加多租户和权限体系。

可实现能力：

- `tenant_id`、`user_id`、`session_id`
- 每个租户独立配置模型、工具、数据库 DSN
- SQL Agent 自动注入租户过滤条件
- 工具权限策略：限制 shell、联网搜索、MCP、SQL 查询等能力
- RBAC：admin / analyst / viewer

企业级难点：

- 数据隔离
- 权限边界
- 工具授权
- 租户级配置管理

## 3. SQL Agent 安全治理

当前项目已有 `sql_query`，这是最适合体现企业级能力的模块之一。

可实现能力：

- SQL 只读模式，禁止 `INSERT`、`UPDATE`、`DELETE`、`DROP`
- SQL AST 解析，避免只靠字符串判断
- 查询超时、最大返回行数、最大扫描量限制
- 敏感字段脱敏：手机号、邮箱、身份证、token
- SQL explain / dry-run
- 自然语言问题到 SQL 的完整审计链路
- Schema allowlist / table allowlist
- 租户级 row-level security

企业级难点：

- 防越权查询
- 防慢查询拖垮数据库
- 敏感数据保护
- LLM 生成 SQL 的可控性

## 4. Agent 工具治理中心

项目已有 `ToolRegistry` 和 MCP 接入，可以继续升级为企业级工具治理层。

可实现能力：

- 工具注册元数据：名称、描述、权限、风险等级、超时、重试策略
- 工具级限流和熔断
- 工具调用缓存
- MCP server 健康检查
- MCP 工具命名冲突处理
- 高风险工具调用前审批
- 工具结果结构化，不只返回字符串

企业级难点：

- 工具权限治理
- 外部依赖稳定性
- 失败隔离
- MCP 多服务管理

## 5. 可观测性与 Trace 回放

企业级 Agent 必须能解释“为什么它这么回答”。

可实现能力：

- LangSmith tracing
- 每轮 LLM 调用记录 prompt、model、latency、tokens
- 工具调用链路图
- 失败原因分类：模型失败、工具失败、权限失败、超时
- `/trace <thread>` 查看一次会话的完整执行链
- Mermaid 自动生成执行图

企业级难点：

- 可调试性
- 可审计性
- 线上问题复盘
- Agent 行为解释

## 6. 配置中心与 Profile

当前配置主要依赖 `.env`，可以扩展为更适合企业部署的配置体系。

可实现能力：

- `~/.askanswer/config.toml`
- 多 profile：dev / prod / company-db
- 每个 profile 配置模型、搜索、数据库、MCP server
- `/profile list`、`/profile use`
- 配置校验和启动诊断
- secret 通过环境变量引用，避免明文落盘

企业级难点：

- 多环境管理
- 密钥管理
- 配置校验
- 团队级共享配置

## 7. 任务队列与异步长任务

当前项目偏交互式问答，可以进一步支持长任务。

可实现能力：

- 长任务后台运行
- `/jobs` 查看任务
- `/cancel <job_id>` 取消任务
- 工具调用超时后可恢复
- SQL 大查询异步执行
- 搜索、文件读取、数据库查询并行 fan-out

企业级难点：

- 长任务生命周期管理
- 任务取消与恢复
- 并发控制
- 用户体验一致性

## 8. 评测与回归测试体系

Agent 项目需要可重复评测，不能只靠人工试用。

可实现能力：

- `evals/` 数据集
- 意图识别准确率测试
- SQL 生成正确率测试
- 工具选择准确率测试
- 回答引用完整性测试
- graph golden snapshot，防止误改图结构
- CI 中跑轻量 mock eval

企业级难点：

- Agent 行为回归
- 质量度量
- 变更风险控制
- 自动化验收

## 9. 安全沙箱与命令执行策略

项目已有 shell 人机确认，可以继续加强为企业级命令执行策略。

可实现能力：

- 命令风险评分
- denylist + allowlist
- 工作目录限制
- 输出大小限制
- secret 检测，防止 `.env` 内容进入模型上下文
- shell 执行审计
- 命令 diff 预览

企业级难点：

- 命令执行安全
- 数据泄露防护
- 高风险操作拦截
- 审批和追责

## 10. Web 管理台

当前已有 `web/` 静态目录，可以扩展成企业管理台。

可实现能力：

- 会话列表
- trace 查看
- 工具调用审计
- token 成本统计
- MCP server 管理
- profile 配置管理
- SQL 查询历史与脱敏结果展示

企业级难点：

- 从 CLI 工具升级为可运营产品
- 可视化治理
- 团队协作
- 管理员视角

## 推荐落地路线

建议优先选择和当前代码最贴合的路线：

1. 持久化 + `/threads` + `/resume`
2. 审计日志 + token/成本统计
3. SQL Agent 安全治理
4. 工具治理中心
5. Web 管理台

这条路线不会变成硬凑功能，也能很好体现企业级 Agent 系统的真实复杂度。
