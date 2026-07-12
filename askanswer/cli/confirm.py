# HITL 确认菜单：把图挂起时的 interrupt 载荷渲染成上下键选择菜单。
#
# 每个 ``_prompt_*`` 返回一个 resume 值（dict），由 stream.py 经 Command(resume=...) 续跑。
# 按 interrupt 载荷的 ``type`` 分发：clarify / fs_write / paid / shell（兜底）。
from __future__ import annotations

import json

from ..tools import gen_shell_command_spec
from ..ui_select import is_interactive, select_option
from .text import _truncate
from .theme import C


def _prompt_confirmation(payload) -> dict:
    """按确认类型分发到对应的 CLI 确认菜单。"""
    data = payload if isinstance(payload, dict) else {}
    confirm_type = data.get("type") or ""
    if confirm_type == "clarify":
        return _prompt_clarification(data)
    if confirm_type == "confirm_fs_write":
        return _prompt_fs_write_confirmation(data)
    if confirm_type == "confirm_external_api_paid":
        return _prompt_paid_confirmation(data)
    return _prompt_shell_confirmation(payload)


def _prompt_clarification(data: dict) -> dict:
    """通用澄清菜单：TTY 弹 ui_select 收集选择；非 TTY 直接取默认项（不阻塞非交互流程）。

    返回 ``{"index", "text"}`` 作为 resume 值：``index`` 是选中项（手动输入项排在最后，
    其 index == 选项数），``text`` 仅命中手动输入项时为用户键入内容。
    """
    labels = [str(x) for x in (data.get("labels") or [])]
    default_index = int(data.get("default_index") or 0)
    has_free_text = bool(data.get("free_text"))
    # 非交互或无任何可选项：按约定回默认项，行为与未澄清时一致。
    if not is_interactive() or (not labels and not has_free_text):
        return {"index": default_index, "text": None}

    print()
    print(f"  {C.ORANGE}⏸{C.RESET}  {C.BOLD}{data.get('prompt') or '需要澄清'}{C.RESET}")
    idx, text = select_option(
        labels,
        prompt="选择（↑/↓ 导航 · Enter 确认）：",
        default=default_index,
        free_input_label=data.get("free_text_label") if has_free_text else None,
        free_input_prompt=data.get("free_text_prompt") or "请输入：",
    )
    return {"index": idx, "text": text}


def _prompt_fs_write_confirmation(data: dict) -> dict:
    """文件写入确认菜单：展示路径、大小变化、diff 预览。"""
    path = data.get("path") or "(unknown)"
    exists = data.get("exists", False)
    size_after = data.get("size_after") or 0
    diff_text = data.get("diff") or ""
    preview_text = data.get("preview") or ""

    print()
    print(f"  {C.ORANGE}⏸{C.RESET}  {C.BOLD}需要确认文件写入{C.RESET}")
    print(f"    {C.DIM}路径：{C.RESET}{C.CYAN}{path}{C.RESET}")
    mode = "覆盖已有文件" if exists else "新建文件"
    print(f"    {C.DIM}操作：{C.RESET}{mode}，写入后 {size_after} 字节")

    if diff_text:
        print()
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                print(f"    {C.GREEN}{line}{C.RESET}")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"    {C.RED}{line}{C.RESET}")
            elif line.startswith("@@"):
                print(f"    {C.CYAN}{line}{C.RESET}")
            else:
                print(f"    {C.DIM}{line}{C.RESET}")
    elif preview_text:
        print()
        for line in preview_text.splitlines():
            print(f"    {C.DIM}{line}{C.RESET}")

    idx, _ = select_option(
        ["写入", "取消"],
        prompt="选择操作（↑/↓ 导航 · Enter 确认）：",
        default=1,
    )
    return {"approve": idx == 0}


def _prompt_paid_confirmation(data: dict) -> dict:
    """付费外部 API 调用确认菜单。"""
    tool_name = data.get("tool") or "(unknown)"
    source = data.get("source") or ""
    tool_args = data.get("args") or {}

    print()
    print(f"  {C.ORANGE}⏸{C.RESET}  {C.BOLD}需要确认付费工具调用{C.RESET}")
    print(f"    {C.DIM}工具：{C.RESET}{C.CYAN}{tool_name}{C.RESET}")
    if source:
        print(f"    {C.DIM}来源：{C.RESET}{source}")
    if tool_args:
        args_str = json.dumps(tool_args, ensure_ascii=False, default=str)
        print(f"    {C.DIM}参数：{C.RESET}{_truncate(args_str, 72)}")
    print(f"    {C.GOLD}⚠ 此调用可能产生费用{C.RESET}")

    idx, _ = select_option(
        ["调用", "取消"],
        prompt="选择操作（↑/↓ 导航 · Enter 确认）：",
        default=1,
    )
    return {"approve": idx == 0}


def _prompt_shell_confirmation(payload) -> dict:
    """用上下选项菜单提示用户确认 shell 命令；选项含 执行 / 取消 / 补充说明后重新生成。"""
    data = payload if isinstance(payload, dict) else {}
    command = data.get("command") or str(payload)
    explanation = data.get("explanation") or ""
    instruction = data.get("instruction") or ""

    # 循环：用户选 “补充说明后重新生成” 时，重新生成命令并再次询问
    while True:
        print()
        print(f"  {C.ORANGE}⏸{C.RESET}  {C.BOLD}需要确认 Shell 命令{C.RESET}")
        print(f"    {C.DIM}命令：{C.RESET}{C.CYAN}{command}{C.RESET}")
        if explanation:
            print(f"    {C.DIM}说明：{C.RESET}{explanation}")
        # 默认光标停在“取消”：避免误回车直接执行高风险命令。
        idx, _ = select_option(
            ["执行", "取消", "补充说明后重新生成"],
            prompt="选择操作（↑/↓ 导航 · Enter 确认）：",
            default=1,
        )
        if idx == 0:
            return {"approve": True, "command": command, "explanation": explanation}
        if idx == 2:
            more = _read_more_prompt()
            if not more:
                print(f"    {C.DIM}未输入补充说明，保持原命令。{C.RESET}")
                continue
            combined = (
                f"{instruction}\n补充说明：{more}".strip()
                if instruction else more
            )
            try:
                new_command, new_explanation = gen_shell_command_spec(combined)
            except Exception as exc:
                print(f"    {C.RED}生成失败：{C.RESET}{exc}")
                continue
            if not new_command:
                print(f"    {C.RED}未能生成有效命令，保持原命令。{C.RESET}")
                continue
            command = new_command
            explanation = new_explanation
            instruction = combined
            continue
        # idx == 1（取消）或 CANCELLED（Esc/Ctrl-C）一律视为不执行
        return {"approve": False, "command": command, "explanation": explanation}


def _read_more_prompt() -> str:
    """让用户输入“补充说明”一行，Ctrl-C/D 视为放弃补充。"""
    try:
        return input(f"    {C.GOLD}补充说明:{C.RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
