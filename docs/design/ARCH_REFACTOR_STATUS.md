# 架构重构落地状态（对照《架构重构设计文档》）

> 约束：**项目名与包名保持 `askanswer` 不变**。设计文档中的 `mazal/...` 路径一律映射到
> `askanswer/...`，不做任何改名。

## 1. P1–P14 逐条状态

| # | 问题 | 现状核对 | 本批处理 | 状态 |
|---|---|---|---|---|
| P1 | 导入时副作用 | `load.py` 导入即 `bootstrap_environ()` 改写环境变量、建 Tavily 客户端；默认模型标签导入时固化 | 去掉导入时装载；入口 `ensure_environ()`；客户端/模型标签惰性；工具限额运行时读取 | ✅ 主要部分完成 |
| P2 | 全局可变单例 | `load.model`、`get_registry()`、`get_persistence()`、intent 注册表均为模块级单例 | 未动（见 §3 第 2 批） | ⏳ |
| P3 | 配置靠改写 `os.environ` | settings.json 被展开成 env，各模块分散 `os.environ.get` | 导入不再改写；仍以 env 作为配置载体 | 🟡 部分 |
| P4 | 状态臃肿 | `SearchState` 混有 `search_query/search_results/file_path`，`step` 为自由字符串 | 未动（见 §3 第 2 批） | ⏳ |
| P5 | 扩展点不彻底 | LLM 分类提示写死 6 个 intent，漏了 research / decision | handler 声明 `description`，提示由注册表生成 | ✅ |
| P6 | 进度提示混进历史 | understand/sorcery 追加提示 `AIMessage`；`_finalize` 重复追加答案 | 全部移除；进度由 CLI 按节点 update 渲染 | ✅ |
| P7 | 评估靠解析文本 | search 评估找“评分：re_search” | `with_structured_output(SearchVerdict)` | ✅ |
| P8 | 工具内藏 LLM 调用 | `read_file` 内调模型摘要；shell 规划内调模型生成命令 | 工具/规划不再调 LLM；CLI 重新生成走 `model_for` | ✅ |
| P9 | HITL 恢复重复执行 | tools 节点先跑普通工具再 `interrupt()`；多确认项会重放前面已批准的动作 | `confirm_plan → ask_approval ⟲ → tools` | ✅（含回归测试） |
| P10 | 外部请求无超时 | `check_weather` 的 `urlopen` 无 timeout；Tavily 用默认值 | 统一显式超时 | ✅ |
| P11 | 持久化职责过多 | `delete_thread` 按表名直删 checkpointer 内部表 | 改用 `SqliteSaver.delete_thread` | 🟡 删除路径已修；拆 repos 未做 |
| P12 | 流式未按节点过滤 | runner 不看 `langgraph_node` | 只放行 `answer` 节点 token | ✅ |
| P13 | 工具标签与技能名混用 | 只有 tags；权限靠字符串相等 | 新增 `risk` / `capability`，注册时强制高风险工具声明确认类；`ASKANSWER_DENY_TOOLS` | ✅ |
| P14 | 目录平铺 | 包内 30+ 顶层模块 | 未动（见 §3 第 3 批） | ⏳ |

## 2. §7.4 不可绕过的不变量

| 不变量 | 状态 |
|---|---|
| WRITE/EXEC/PAID 工具不声明审批无法注册 | ✅ `ToolDescriptor.__post_init__`，有测试 |
| act 执行审批项前二次检查 | ✅ 各 handler 的 `apply()` 内重新校验 |
| 会 interrupt 的节点无副作用 | ✅ `clarify` / `ask_approval`，有回归测试 |
| 子 Agent 拿不到需审批的工具 | ✅ 现状即如此（子图不绑定 shell / write_file） |
| 非 TTY 审批默认拒绝 | ✅ 现状即如此（`select_option` 非 TTY 返回默认项“取消”） |
| 审计落库前脱敏 | 🟡 `redact_audit_args` 只用于付费 API 参数，其余仅记录 path/command |

## 3. 后续批次（按依赖顺序，每批 pytest 全绿）

1. **第 2 批：状态与组装根**（P2、P4）
   - `SearchState` 收敛为通用字段 + `turn_slots: dict`，`step` 改为枚举；`search_query/file_path` 迁入 slots，旧 checkpoint 读取时迁移。
   - 新增 `askanswer/runtime/container.py`：显式组装 ModelHub / ToolRegistry / Store，节点改为 `make_xxx_node(container)` 工厂；单测用 `FakeModelFactory`，不再 monkeypatch 模块全局。
   - 类型化 `Settings`（Pydantic）只在 loader 里读 env，其余模块通过注入取配置（完成 P3）。
2. **第 3 批：分层目录**（P14、P11 剩余）
   - 在 `askanswer/` 下建立 `core/ config/ llm/ graph/ tools/ safety/ storage/ observability/ runtime/ interfaces/`，逐个模块迁移；**包名仍为 askanswer**。
   - `persistence.py` 拆为 `storage/db.py`（连接+迁移 SQL 文件）+ `storage/repos/{threads,audit,labels}.py` + `Store` 门面。
   - 审计 / 用量 / 遥测改为订阅事件总线（设计文档 3.1-7）。
3. **第 4 批：图扁平化**（D1）
   - 父图 + react 子图合并为单层 7 节点（route / clarify / reason / plan_approvals / ask_approval / act / review），加 Mermaid golden test。

## 4. 验证

```
pip install -e '.[dev]'
pytest -q          # 本批结束时全部通过
ruff check askanswer tests
python -m askanswer --graph   # 无任何环境变量也能导出
```
