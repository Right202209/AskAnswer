"""CLI 确认界面的「补充说明后重新生成命令」辅助。

这是接入层的交互功能，不是工具：工具与确认规划都不调用 LLM（架构文档 P8）。
这里走 ``model_for(ROLE_ANSWER)``，因此仍经过模型路由、回退链与用量审计。
"""

from __future__ import annotations

import platform

from askanswer.llm.routing import ROLE_ANSWER, model_for
from askanswer.tools.builtin import clean_command


def gen_shell_command_spec(instruction: str) -> tuple[str, str]:
    """根据自然语言指令生成 (命令, 解释)。"""
    current_os = f"{platform.system()} {platform.release()} {platform.machine()}"
    prompt = (
        f"根据以下用户指令生成在 {current_os} 环境下可直接执行的单条 shell 命令。\n"
        "严格按以下两行格式输出，不要 markdown、不要多余内容：\n"
        "命令：<shell command>\n"
        "说明：<一句话解释它做什么>\n\n"
        f"用户指令：{instruction}"
    )
    resp = model_for(ROLE_ANSWER).invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    return parse_command_spec(str(text))


def parse_command_spec(text: str) -> tuple[str, str]:
    command = ""
    explanation = ""
    for line in text.splitlines():
        stripped = line.strip()
        for prefix in ("命令：", "命令:"):
            if stripped.startswith(prefix):
                command = stripped[len(prefix):].strip()
        for prefix in ("说明：", "说明:"):
            if stripped.startswith(prefix):
                explanation = stripped[len(prefix):].strip()
    return clean_command(command or text), explanation
