"""Shell tool: LLM-synthesised commands behind a dangerous-pattern gate.

``gen_shell_commands_run`` is registered with ``confirmation_class="shell"``: the
react subgraph plans the command, checks :data:`_DANGEROUS_PATTERNS`, asks the
human, re-checks the (possibly edited) command and only then calls
:func:`execute_shell_command` (see :mod:`askanswer.core.confirmations`).
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess

from langchain_core.tools import tool

from ..llm.proxy import model

# 高风险命令模式列表：在“生成命令”和“用户编辑命令”两个时机都会被检查
_DANGEROUS_PATTERNS = [
    (r"\brm\b", "rm 删除命令"),
    (r"\brmdir\b", "rmdir 目录删除"),
    (r"\bshutdown\b", "shutdown 关机"),
    (r"\breboot\b", "reboot 重启"),
    (r"\bhalt\b", "halt 停机"),
    (r"\bpoweroff\b", "poweroff 关机"),
    (r"\bmkfs\w*\b", "mkfs 格式化"),
    (r"(?:^|\s)dd\s+if=", "dd 底层磁盘写入"),
    (r":\s*\(\)\s*\{", "fork bomb"),
    (r"\bsudo\b", "sudo 提权"),
    (r"(?<!>)>(?![>&])", "重定向覆盖 `>`（会覆盖文件）"),
    (r"\bkill\s+-9\b", "kill -9 强制终止"),
    (r"\bchmod\s+-R\b", "chmod -R 递归改权限"),
    (r"\bchown\s+-R\b", "chown -R 递归改属主"),
    (r">\s*/dev/sd", "写入磁盘设备"),
    (r"\bmv\s+\S+\s+/\b", "mv 到根目录"),
]


def _check_dangerous(command: str) -> str | None:
    """检查命令是否命中危险模式，命中返回中文描述，否则返回 None。"""
    for pattern, desc in _DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            return desc
    return None


def _clean_command(raw: str) -> str:
    """清理 LLM 输出里偶尔残留的 ``` 代码块围栏与开头的 $ 提示符。"""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s.lstrip("$").strip()


def gen_shell_command_spec(instruction: str) -> tuple[str, str]:
    """根据自然语言指令生成（命令, 解释）二元组。

    会把当前操作系统类型告诉 LLM，让它生成对应平台可直接执行的单条命令。
    """
    # 把当前 OS 信息拼到 prompt 里，避免 mac/linux/windows 命令混用
    current_os = platform.system() + " " + platform.release() + " " + platform.machine()
    prompt = (
        f"根据以下用户指令生成在 {current_os} 环境下可直接执行的单条 shell 命令。\n"
        f"严格按以下两行格式输出，不要 markdown、不要多余内容：\n"
        f"命令：<shell command>\n"
        f"说明：<一句话解释它做什么>\n\n"
        f"用户指令：{instruction}"
    )
    resp = model.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    command = ""
    explanation = ""
    # 逐行扫描；同时兼容中文和英文冒号
    for line in text.splitlines():
        stripped = line.strip()
        for prefix in ("命令：", "命令:"):
            if stripped.startswith(prefix):
                command = stripped[len(prefix):].strip()
                break
        for prefix in ("说明：", "说明:"):
            if stripped.startswith(prefix):
                explanation = stripped[len(prefix):].strip()
                break
    # 兜底：如果两行格式都没匹配上，就把整段当成命令再清理一遍
    command = _clean_command(command or text)
    return command, explanation


def execute_shell_command(command: str, shell: bool = False) -> str:
    """执行 shell 命令并把 stdout/stderr/返回码包成易读的文本返回。

    捕获 stdout/stderr，并对单次调用设置 30 秒超时，适合 agent 的非交互执行路径。
    """
    popen_args: str | list[str]
    # shell=False 时用 shlex 拆分参数，避免命令注入；shell=True 时直接交给 /bin/sh
    if shell:
        popen_args = command
    else:
        try:
            popen_args = shlex.split(command)
        except ValueError as exc:
            return f"命令解析失败：{exc}\n原始命令：{command}"

    try:
        process = subprocess.Popen(
            popen_args,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # 30s 超时：避免误执行长时间运行的命令导致整个图卡住
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        # 超时后必须 kill + 收尸，否则会有僵尸进程
        process.kill()
        process.communicate()
        return f"命令执行超时（>30s）：{command}"
    except FileNotFoundError as exc:
        return f"命令未找到：{exc}"
    except Exception as exc:
        return f"命令执行失败：{exc}"

    ok = process.returncode == 0
    parts = [
        f"命令：{command}",
        f"返回码：{process.returncode}（{'成功' if ok else '失败'}）",
    ]
    # 输出截断：每路 stream 各 64 KB 上限，防止单条命令的大输出把 prompt 撑爆。
    if stdout:
        parts.append(f"stdout:\n{_truncate_stream(stdout.strip(), 'stdout')}")
    if stderr:
        parts.append(f"stderr:\n{_truncate_stream(stderr.strip(), 'stderr')}")
    return "\n".join(parts)


# 单路 stdout/stderr 的文本上限；超出则截断并附说明，避免拖垮 LLM prompt 与 checkpoint。
# 调用期读取 ASKANSWER_SHELL_OUTPUT_MAX_BYTES；模块级变量非 None 时优先（测试覆盖用）。
_SHELL_OUTPUT_MAX_BYTES: int | None = None
_DEFAULT_SHELL_OUTPUT_MAX_BYTES = 64 * 1024


def shell_output_max_bytes() -> int:
    if _SHELL_OUTPUT_MAX_BYTES is not None:
        return _SHELL_OUTPUT_MAX_BYTES
    try:
        return int(
            os.environ.get("ASKANSWER_SHELL_OUTPUT_MAX_BYTES") or _DEFAULT_SHELL_OUTPUT_MAX_BYTES
        )
    except ValueError:
        return _DEFAULT_SHELL_OUTPUT_MAX_BYTES


def _truncate_stream(text: str, label: str) -> str:
    """对 shell 子进程单路输出做硬截断；命中上限时附带省略字符数提示。"""
    limit = shell_output_max_bytes()
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n[{label} 已截断，省略 {omitted} 字符]"


# 对外暴露的危险检查别名，供确认执行层复用。
check_dangerous = _check_dangerous


@tool
def gen_shell_commands_run(instruction: str) -> str:
    """
    根据用户指令生成并执行 shell 命令。
    高风险操作（rm、shutdown、重定向覆盖等）会被自动拦截；
    执行前会暂停图流程，向用户展示命令与说明，用户确认后才真正执行。
    """
    # 正常情况下，此工具的 tool_call 会被 tools_node 拦截并走人机确认流程；
    # 这里仅作为“脱离图流程被直接调用”时的兜底，不自动执行、要求使用图入口。
    command, explanation = gen_shell_command_spec(instruction)
    if not command:
        return "未能生成有效的 shell 命令"
    danger = check_dangerous(command)
    if danger:
        return f"已拦截高风险命令（{danger}）：{command}"
    return (
        "此工具需在图流程中运行以获得用户确认。\n"
        f"拟执行命令：{command}\n说明：{explanation or '(无)'}"
    )
