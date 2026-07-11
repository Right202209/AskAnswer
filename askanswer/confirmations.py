"""确认类工具（``confirmation_class``）的 HITL 执行协议。

``pending_shell`` 模式的泛化：注册表里声明的三个确认类（shell / fs_write /
external_api_paid）各自实现同一套四步协议，react 子图按类分发，不再写死 shell：

1. ``plan()``   —— ``interrupt()`` **之前**把“要执行的具体动作”物化成 dict，写入
   ``state.pending_confirmations`` 由父图 checkpointer 持久化。resume 后直接复用，
   绝不重新调用 LLM（重调不仅多花 token，还可能生成不同的动作导致用户白确认一次）。
2. ``gate()``   —— 弹确认框之前的安全闸门（shell 危险命令检查、fs_write 敏感路径
   检查等），命中即拦截、根本不打扰用户。
3. ``interrupt_payload()`` —— 交给 CLI 渲染的载荷，``type`` 固定为
   ``confirm_<class>``，CLI 据此选择对应的确认菜单。
4. ``apply()``  —— 用户批准后执行。内部必须重新过一遍安全闸门（用户可能编辑过
   内容，如 shell 命令），这是刻意的双重检查，不要省略。

新增确认类 = 在这里实现一个 handler 并加入 ``_HANDLERS``，不要在
``_react_internals`` / ``cli`` 里加 if/elif 分支（对齐 intent 的扩展方式）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tools import (
    check_dangerous,
    describe_write,
    execute_shell_command,
    gen_shell_command_spec,
    validate_write_path,
)


@dataclass(frozen=True)
class ConfirmationOutcome:
    """``apply()`` 的返回值：回填 ToolMessage 的内容 + 审计所需元数据。"""
    content: str                       # ToolMessage 的正文
    approved: bool                     # 是否真正执行了动作
    error: str | None = None           # 审计 error 字段（拦截原因 / 执行异常）
    audit_args: dict = field(default_factory=dict)  # 审计 args_summary 的原始字典


# ── 决策解析（CLI resume 值 → 结构化结果） ─────────────────────────────

def parse_decision(decision: Any, fallback_command: str) -> tuple[bool, str]:
    """把 CLI 的 resume 值统一解析成 (approved, command)，shell 类专用（命令可编辑）。"""
    # 直接传 True：批准，沿用旧命令
    if decision is True:
        return True, fallback_command
    if isinstance(decision, dict):
        # 字典形式：兼容 approve / value 两个键名
        approve = decision.get("approve")
        if approve is None:
            approve = decision.get("value")
        cmd = decision.get("command") or fallback_command
        if isinstance(approve, bool):
            return approve, cmd
        return _truthy(approve), cmd
    return _truthy(decision), fallback_command


def parse_approval(decision: Any) -> bool:
    """只关心批准与否的确认类（fs_write / external_api_paid）用的简化解析。"""
    approved, _ = parse_decision(decision, fallback_command="")
    return approved


def _truthy(value: Any) -> bool:
    """把字符串/None 等输入转换成布尔，用来兼容人类敲的 y/yes/1 等。"""
    if value is None:
        return False
    return str(value).strip().lower() in {"y", "yes", "true", "1", "approve"}


# ── shell：生成命令 → 危险检查 → 确认 → 再检查 → 执行 ─────────────────

class ShellConfirmation:
    clazz = "shell"

    def plan(self, tool_call: dict) -> dict:
        # 工具入参里取自然语言指令；instruction 优先，input 兜底
        args = tool_call.get("args") or {}
        instruction = (args.get("instruction") or args.get("input") or "").strip()
        if not instruction:
            return {"command": "", "explanation": "未提供 shell 指令", "instruction": ""}
        try:
            command, explanation = gen_shell_command_spec(instruction)
        except Exception as exc:
            # 生成失败也返回结构化 payload，让 gate 据此拦截并给出友好错误
            return {
                "command": "",
                "explanation": f"生成 shell 命令失败：{exc}",
                "instruction": instruction,
            }
        return {"command": command, "explanation": explanation, "instruction": instruction}

    def gate(self, payload: dict) -> str | None:
        command = payload.get("command") or ""
        if not command:
            return payload.get("explanation") or "未能生成有效的 shell 命令"
        # 用户确认前先做一次危险命令检查，避免诱导用户点 y 后造成损失
        danger = check_dangerous(command)
        if danger:
            return f"已拦截高风险命令（{danger}）：{command}"
        return None

    def interrupt_payload(self, payload: dict) -> dict:
        return {
            "type": "confirm_shell",
            "command": payload.get("command") or "",
            "explanation": payload.get("explanation") or "",
            "instruction": payload.get("instruction") or "",
        }

    def audit_args(self, payload: dict) -> dict:
        return {"command": payload.get("command") or ""}

    def apply(self, payload: dict, decision: Any, tool_call: dict) -> ConfirmationOutcome:
        approved, command = parse_decision(
            decision, fallback_command=payload.get("command") or ""
        )
        if not approved:
            return ConfirmationOutcome(
                content=f"已取消执行：{command}",
                approved=False,
                audit_args={"command": command},
            )
        # 用户可能修改了命令，必须重新做危险检查
        danger = check_dangerous(command)
        if danger:
            return ConfirmationOutcome(
                content=f"已拦截高风险命令（{danger}）：{command}",
                approved=False,
                error=f"dangerous command: {danger}",
                audit_args={"command": command},
            )
        content = execute_shell_command(command)
        return ConfirmationOutcome(
            content=content, approved=True, audit_args={"command": command}
        )


# ── fs_write：敏感路径/大小校验 → diff 预览 → 确认 → 再校验 → 落盘 ────

class FsWriteConfirmation:
    clazz = "fs_write"

    def plan(self, tool_call: dict) -> dict:
        args = tool_call.get("args") or {}
        path = str(args.get("path") or "").strip()
        content = "" if args.get("content") is None else str(args.get("content"))
        error = validate_write_path(path, content)
        if error:
            return {"path": path, "error": error}
        # 只存派生信息（diff/预览/大小）；content 本体已在消息历史的 tool_call args 里
        return {"path": path, **describe_write(path, content)}

    def gate(self, payload: dict) -> str | None:
        # plan 阶段的校验错误由通用流程按 payload["error"] 拦截，这里无需额外闸门
        return None

    def interrupt_payload(self, payload: dict) -> dict:
        return {
            "type": "confirm_fs_write",
            "path": payload.get("path") or "",
            "exists": bool(payload.get("exists")),
            "size_before": payload.get("size_before") or 0,
            "size_after": payload.get("size_after") or 0,
            "diff": payload.get("diff") or "",
            "preview": payload.get("preview") or "",
        }

    def audit_args(self, payload: dict) -> dict:
        return {"path": payload.get("path") or ""}

    def apply(self, payload: dict, decision: Any, tool_call: dict) -> ConfirmationOutcome:
        from pathlib import Path

        path = payload.get("path") or ""
        audit = {"path": path}
        if not parse_approval(decision):
            return ConfirmationOutcome(
                content=f"已取消写入：{path}", approved=False, audit_args=audit
            )
        args = tool_call.get("args") or {}
        content = "" if args.get("content") is None else str(args.get("content"))
        # 双重检查：确认框弹出到批准之间文件系统可能变化（如路径变成目录/符号链接）
        error = validate_write_path(path, content)
        if error:
            return ConfirmationOutcome(
                content=error, approved=False, error=error, audit_args=audit
            )
        try:
            target = Path(path).expanduser().resolve()
            existed = target.is_file()
            target.write_text(content, encoding="utf-8")
        except Exception as exc:
            message = f"写入失败：{exc}"
            return ConfirmationOutcome(
                content=message, approved=True, error=str(exc), audit_args=audit
            )
        size = len(content.encode("utf-8", errors="replace"))
        mode = "覆盖" if existed else "新建"
        return ConfirmationOutcome(
            content=f"已写入 {path}（{mode}，{size} 字节）",
            approved=True,
            audit_args=audit,
        )


# ── external_api_paid：展示工具与参数 → 确认 → 直接调用 ───────────────

class PaidApiConfirmation:
    """付费外部 API 的通用确认：规划无需 LLM，批准后按注册表原样调用。

    内置工具目前没有此类；MCP / 用户自定义工具把 ``confirmation_class`` 标成
    ``external_api_paid`` 即可获得“先确认再调用”的闸门。
    注意：批准后是 ``descriptor.tool.invoke(args)`` 直调，不经过 ToolNode ——
    依赖 ``ToolRuntime`` 注入的工具暂不支持挂到这个确认类上。
    """

    clazz = "external_api_paid"

    def plan(self, tool_call: dict) -> dict:
        source = ""
        descriptor = self._descriptor(tool_call.get("name") or "")
        if descriptor is not None:
            source = descriptor.source
        return {
            "tool": tool_call.get("name") or "",
            "args": dict(tool_call.get("args") or {}),
            "source": source,
        }

    def gate(self, payload: dict) -> str | None:
        return None

    def interrupt_payload(self, payload: dict) -> dict:
        return {
            "type": "confirm_external_api_paid",
            "tool": payload.get("tool") or "",
            "args": payload.get("args") or {},
            "source": payload.get("source") or "",
        }

    def audit_args(self, payload: dict) -> dict:
        return {"tool": payload.get("tool") or "", "args": payload.get("args") or {}}

    def apply(self, payload: dict, decision: Any, tool_call: dict) -> ConfirmationOutcome:
        name = payload.get("tool") or tool_call.get("name") or ""
        audit = {"tool": name, "args": payload.get("args") or {}}
        if not parse_approval(decision):
            return ConfirmationOutcome(
                content=f"已取消调用付费工具：{name}", approved=False, audit_args=audit
            )
        descriptor = self._descriptor(name)
        if descriptor is None:
            message = f"未知工具：{name}"
            return ConfirmationOutcome(
                content=message, approved=False, error=message, audit_args=audit
            )
        try:
            result = descriptor.tool.invoke(payload.get("args") or {})
        except Exception as exc:
            # 已批准但执行失败：包成友好字符串回填，让 LLM 看到错误并自行决策
            return ConfirmationOutcome(
                content=f"工具 {name} 调用失败：{exc}",
                approved=True,
                error=str(exc),
                audit_args=audit,
            )
        return ConfirmationOutcome(content=str(result), approved=True, audit_args=audit)

    @staticmethod
    def _descriptor(name: str):
        # 延迟 import：registry 在 seed 阶段会 import tools，避免加载顺序问题
        from .registry import get_registry

        return get_registry().get(name)


_HANDLERS = {
    handler.clazz: handler
    for handler in (ShellConfirmation(), FsWriteConfirmation(), PaidApiConfirmation())
}


def get_confirmation_handler(clazz: str | None):
    """按确认类取 handler；未接入的类（含 "none"/None）返回 None。"""
    if not clazz:
        return None
    return _HANDLERS.get(clazz)


def supported_confirmation_classes() -> frozenset[str]:
    return frozenset(_HANDLERS)
