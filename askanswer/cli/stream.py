# 查询执行与实时渲染：消费 runner 的一腿事件流，边跑边渲染答案 + 处理 HITL。
#
# ``stream_query`` 是 CLI 的问答主入口：
# - 事件来源统一走 ``runner.stream_leg``（与 HTTP server 同一份事件流，不再各造一份）；
# - token 事件走 rich.Live 实时渲染，node 事件打 ⏺ 标记，interrupt 事件弹确认菜单；
# - 记账（audit begin/end、telemetry 根 span、thread_meta 落库）仍由 CLI 按「整轮」口径
#   自持（runner.stream_leg 只出事件、不记账），因此多腿一轮的审计粒度与拆分前一致。
from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule

from .. import runner
from ..audit import begin_run, end_run, flush_pending
from ..load import current_model_label
from ..persistence import get_persistence
from ..schema import ContextSchema
from ..ui_spinner import Spinner
from .confirm import _prompt_confirmation
from .progress import _marker, _phase_text, _render_node_update_safely
from .render import render_answer
from .theme import _console


def _telemetry_span(name: str, **attrs):
    """返回一个 telemetry span 上下文管理器；未启用时是零开销 no-op。"""
    from .. import telemetry

    return telemetry.span(name, **attrs)


def _open_root_span(thread_id: str, tenant_id: str | None):
    """为一轮请求开根 span；未启用返回 None。异常不阻断主流程。"""
    try:
        from .. import telemetry

        return telemetry.open_span(
            "askanswer.query", thread_id=thread_id, tenant_id=tenant_id or ""
        )
    except Exception:
        return None


def _close_root_span(handle) -> None:
    """关闭根 span；handle 为 None 或未启用时 no-op。"""
    if handle is None:
        return
    try:
        from .. import telemetry

        telemetry.close_span(handle)
    except Exception:
        pass


def _runtime_context() -> ContextSchema:
    """从环境变量构造一份 ContextSchema 传给图（env→context 的统一口径在 runner，CLI/HTTP 共用）。"""
    return runner.runtime_context_from_env()


def stream_query(
    app,
    query: str,
    thread_id: str,
    runtime_context: ContextSchema | None = None,
) -> str:
    """跑一次完整的图调用：spinner 提示等待 + 实时流式渲染答案 + 处理 HITL。"""
    final_answer = ""
    config = runner.thread_config(thread_id)
    context = runtime_context or _runtime_context()
    graph_input: object = runner.query_input(query)
    audit_tokens = begin_run(thread_id, tenant_id=context.tenant_id)
    # 整轮请求作为 telemetry 根 span；未启用时 handle 为 None，收尾时 no-op。
    telemetry_span = _open_root_span(thread_id, context.tenant_id)
    # 缓存一次最终状态，避免 finally 与下文 meta 持久化各 get_state 一次
    final_state_values: dict | None = None

    print()
    spinner = Spinner("理解意图…")
    spinner.start()

    # rich.Live 在 answer 节点的 LLM token 流到达时启动，承担实时渲染。
    # streamed 标记用来告诉调用者“最终答案已经在屏上了，不要再 render_answer 一次”。
    live_state = {"live": None, "buf": "", "in_tool": False, "streamed": False}

    try:
        while True:
            interrupt_payload = None
            # 一腿 = 从一次图输入跑到 interrupt 或完成；runner 负责节点计时与 interrupt 兜底。
            for ev in runner.stream_leg(
                app, graph_input, config=config, context=context,
            ):
                if ev.kind == runner.EVENT_TOKEN:
                    _render_token(ev.text, spinner, live_state)
                elif ev.kind == runner.EVENT_TOOL:
                    # LLM 正在“规划工具调用”：标记 phase，下一段正文来时重置 buffer。
                    live_state["in_tool"] = True
                elif ev.kind == runner.EVENT_NODE:
                    final_answer = _on_node_update(
                        ev.node, ev.data or {}, ev.elapsed or 0.0,
                        final_answer, spinner, live_state,
                    )
                elif ev.kind == runner.EVENT_INTERRUPT:
                    interrupt_payload = ev.data
                elif ev.kind == runner.EVENT_FINAL:
                    if ev.text:
                        final_answer = ev.text

            if interrupt_payload is None:
                break

            # HITL：暂停 spinner 让用户看清楚要确认的命令；resume 后再启动新一轮。
            spinner.stop()
            _close_live(live_state)
            resume_value = _prompt_confirmation(interrupt_payload)
            graph_input = Command(resume=resume_value)
            spinner = Spinner("继续执行…")
            spinner.start()
    finally:
        _close_live(live_state)
        spinner.stop()
        audit_intent = None
        try:
            audit_state = app.get_state(config)
            final_state_values = getattr(audit_state, "values", {}) or {}
            audit_intent = final_state_values.get("intent")
        except Exception:
            pass
        flush_pending(thread_id=thread_id, intent=audit_intent)
        _close_root_span(telemetry_span)
        end_run(audit_tokens)

    # 兜底：若节点流里没拿到 final_answer，从 state 里找最后一条消息内容
    if not final_answer:
        try:
            vals = final_state_values
            if vals is None:
                state = app.get_state({"configurable": {"thread_id": thread_id}})
                vals = getattr(state, "values", {}) or {}
                final_state_values = vals
            final_answer = vals.get("final_answer") or ""
            if not final_answer:
                msgs = vals.get("messages") or []
                if msgs:
                    content = getattr(msgs[-1], "content", "")
                    if isinstance(content, str):
                        final_answer = content
        except Exception:
            pass

    # Live 已经把答案画在屏上了，就只补一个空行做间距；
    # 没走 Live 的（典型场景：模型不支持 stream，或 sorcery 重写答案）才走传统渲染。
    if live_state.get("streamed"):
        print()
    else:
        render_answer(final_answer or "未生成答案。")

    # 持久化线程元数据：每次问答后写一行（首次写入会自动取 preview 前 30 字符做 title）。
    # 失败不影响主流程 —— 用户拿到回答比记账更重要。
    try:
        meta_values = final_state_values
        if meta_values is None:
            meta_state = app.get_state(config)
            meta_values = getattr(meta_state, "values", {}) or {}
        msgs = meta_values.get("messages") or []
        human_count = sum(1 for m in msgs if isinstance(m, HumanMessage))
        preview_text = (query or "").strip().replace("\n", " ")[:80] or None
        get_persistence().upsert_meta(
            thread_id,
            intent=meta_values.get("intent"),
            model_label=current_model_label(),
            preview=preview_text,
            message_count=human_count,
            tenant_id=context.tenant_id,
        )
    except Exception:
        pass

    return final_answer or "未生成答案。"


def _render_token(content: str, spinner: Spinner, live_state: dict) -> None:
    """把 runner 的 token 事件渲染到 rich.Live（从旧 _handle_message_chunk 的正文分支抽出）。

    首次 user-facing token 到达时切到 rich.Live：spinner 让位、Markdown 实时刷新；
    若上一段是“工具规划期”的胡言乱语（in_tool 置位），先清空 buffer 再累积正文。
    """
    if live_state["in_tool"]:
        live_state["buf"] = ""
        live_state["in_tool"] = False

    if live_state["live"] is None:
        # 第一次拿到 user-facing token：让 spinner 让位，启动 Live 渲染。
        # 在 Live 之前插一条 Rule，把"进度 trace"与"答案正文"在视觉上拆开。
        spinner.stop()
        _console.print()
        _console.print(Rule(title="[subtle]Answer[/]", style="muted", align="left"))
        live = Live(
            Padding(Markdown(""), (0, 2)),
            console=_console,
            refresh_per_second=15,
            transient=False,
        )
        live.start()
        live_state["live"] = live
        live_state["streamed"] = True

    live_state["buf"] += content
    live_state["live"].update(Padding(Markdown(live_state["buf"]), (0, 2)))


def _on_node_update(
    node: str,
    update: dict,
    elapsed: float,
    final_answer: str,
    spinner: Spinner,
    live_state: dict,
) -> str:
    """节点完成时被调：打 ⏺ 标记 + 维护 spinner / Live 生命周期。

    返回新的 ``final_answer``（沿用旧的或被节点更新覆盖）。
    """
    # 父图的 "answer" 节点完成（react 子图整段跑完）：把 Live 收尾、答案确权。
    if node == "answer":
        if update.get("final_answer"):
            final_answer = update["final_answer"]
        if live_state["live"] is not None:
            # 用权威 final_answer 替换 Live 的当前内容，防止 buffer 与最终答案有偏差
            if final_answer:
                live_state["live"].update(
                    Padding(Markdown(final_answer), (0, 2))
                )
            _close_live(live_state)
        spinner.freeze_for(lambda: print(_marker("Answer", "完成", elapsed)))
        spinner.transition(_phase_text("sorcery"))
        return final_answer

    # clarify 节点绝大多数轮次无澄清（返回空 update）：完全隐身，不打标记也不切
    # spinner 文案，避免每轮都闪一行「澄清需求…」。
    if node == "clarify" and not update:
        return final_answer

    # 其它节点：一律先暂停 spinner 写一行 ⏺ 标记，再切到下一阶段文案。
    new_final = _render_node_update_safely(
        node, update, final_answer, elapsed, spinner,
    )
    spinner.transition(_phase_text(node))
    return new_final


def _close_live(live_state: dict) -> None:
    """关闭 rich.Live，保留已渲染内容（transient=False）。"""
    live = live_state.get("live")
    if live is not None:
        live.stop()
    live_state["live"] = None
    # buffer 不清，保留给下游 final_answer 兜底用
