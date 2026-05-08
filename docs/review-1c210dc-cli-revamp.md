# Code Review — `1c210dc feat: revamp CLI with prompt_toolkit, spinner, and live token streaming`

> 评审范围：HEAD 提交 `1c210dc`（2026-05-08）
> 涉及文件：`askanswer/cli.py`、`askanswer/ui_input.py`(new)、`askanswer/ui_spinner.py`(new)、`pyproject.toml`

## 概览

把 CLI 输入栏从裸 `input()` 升级到 `prompt_toolkit`（历史/补全/底部状态栏/双 Ctrl-C 退出），同时把流式输出从「每节点一行 ⏺」升级为「等待时 spinner 转圈 + 收到 token 时 rich.Live 实时渲染 Markdown」。新增 `ui_input.py`、`ui_spinner.py`，重写 `cli.py:stream_query` / `interactive_loop`。整体结构清晰、职责拆分合理。

---

## 主要问题（按严重度）

### 1. Live 缓冲区未在关闭时清空（潜在 bug）
- 位置：`cli.py:504-510` `_close_live`
- 现象：当 sorcery 判定 `retry_search`、answer 节点二次运行时：
  - 第二轮第一个 token 到达，`live_state["live"] is None` → 重开 Live
  - 但 `buf` 仍是上一轮完整答案，新 token 被拼接其后 → Live 显示「旧答案 + 新答案」串联错乱
- 建议修复：`_close_live` 同时把 `buf` 置空；或在 answer 分支用权威 `final_answer` 覆盖 Live 后立即清空。

### 2. Live 启动会终结 spinner 线程，sorcery 阶段无 spinner（体验回退）
- 位置：`cli.py:441` `spinner.stop()` + `cli.py:480` `spinner.transition(...)`；`ui_spinner.py:80-84` `transition()`
- 现象：`transition` 只改文案/重置秒表，**不会重启已 join 的后台线程**。从 answer 收尾到 sorcery 完成这段时间，用户看不到任何「转圈」反馈，与本次提交宣称的体验不符。
- 建议修复：`_close_live` 之后显式 `spinner.start()`；或在 `_on_node_update` 的 answer 分支替换为全新 `Spinner` 并 `start`。

### 3. `stream_query` 中 Ctrl-C 直接吞掉整个 REPL
- 位置：`cli.py:1213` `except Exception`
- 现象：`KeyboardInterrupt` 继承自 `BaseException`，会逃逸 REPL 让进程退出。新版 LLM 流可能跑很久，"想取消但不退应用"是常见诉求。
- 建议修复：`stream_query` 调用处补一层 `except KeyboardInterrupt:`，把 spinner/Live 收尾后 `continue`。

### 4. `_handle_message_chunk` 的 `in_tool` 复位时机
- 位置：`cli.py:427-437`
- 现象：只在「下一段 user-facing content」到达时清 buffer。如果模型在同一轮 react 内顺序产出 `content → tool_call_chunks → content`，第一段 content 不会被丢弃。对会多次产出 thinking 文本的模型，会污染最终答案的 Live 渲染。
- 建议修复：`tool_call_chunks` 出现时直接 `buf = ""`（而不是延迟到下一段 content）。

---

## 次要问题

| # | 位置 | 问题 |
|---|---|---|
| 5 | `ui_input.py:160-163` | `cont_prompt` 是 `"│ " + "·  "`（5 列），主 prompt 是 `"│ " + "> "`（4 列），续行光标位置错位 1 列。|
| 6 | `cli.py:452-453` | `Live` 每个 token 都重建 `Markdown(...)` 并 `update`，长答案会有可观 CPU；`refresh_per_second=15` 限制刷新但不限制构造。可加节流（每 N 字符 / 每 50ms 才更新一次）。|
| 7 | `ui_input.py:77-86` | `~/.askanswer/history` 含潜在敏感内容。可在 README 提一句，或为 `/clear` 提供「同时清历史」选项。|
| 8 | `ui_input.py:34-45` | `SLASH_COMMANDS` 不含 `/quit` `/q`，补全菜单看不到这些别名（`handle_command` 仍能识别）。|
| 9 | `ui_input.py:104-106` | `_last_interrupt_ts` 是模块级全局，单进程多 session 时共享状态；当前 CLI 模型下不会出问题，但作为 `ui_input.py` 的对外 API 略不洁。|

---

## 设计上做得好的地方

- `ui_input` / `ui_spinner` 拆成独立模块，`cli.py` 只做编排；循环依赖用「ANSI 常量复制一份」回避，决策合理。
- `SLASH_COMMANDS` 同时驱动 `_SlashCompleter`、`/help` 列表、`/help <cmd>` 详情，单一数据源。
- `_build_status_provider(thread_box)` 用单元素列表把 `thread_id` 传引用，使 toolbar 自动反映 `/clear`/`/resume` 切换 —— 简洁。
- `Spinner._is_tty()` no-op + `freeze_for` 锁机制，覆盖 CI/管道场景，避免污染日志。
- 单次问答 (`args.question`) 和 REPL 都收敛到 `stream_query`，渲染路径统一（`live_state["streamed"]` 决定是否兜底 `render_answer`）。

---

## 验证建议

修完上面 1、2、3 后，人工再跑一遍：
- [ ] 触发 sorcery `retry_search` 的查询，确认 Live 不出现旧答案残留
- [ ] 长 query 中按 Ctrl-C，确认能取消并回到提示符
- [ ] HITL shell 确认前后视觉连续性（spinner 是否衔接）
- [ ] 非 TTY（`echo q | askanswer`）下不出现转圈字符

可补少量单测：mock stdout / 注入伪 chunk，覆盖 `in_tool/buffer 复位`、`freeze_for 抢占`、`stop→start 复用` 等路径。

---

## 总评

方向正确、用户体验提升明显；流式 + 多阶段 + 重试组合下的状态机有 2 个真实漏洞，建议合并/上线前修掉问题 1、2、3。
