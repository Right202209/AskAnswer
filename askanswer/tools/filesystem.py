"""Filesystem tools and their safety gates.

- ``read_file``: sensitive-path / size / existence checks, then markitdown + LLM analysis.
- ``write_file``: registered with ``confirmation_class="fs_write"``; the actual write
  happens in :mod:`askanswer.core.confirmations` after human approval.
  ``validate_write_path`` and ``describe_write`` feed that HITL flow.

Size limits are read from the environment at call time so values loaded lazily
from ``settings.json`` take effect.
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path

from langchain_core.tools import tool

from ..llm.proxy import model


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


@tool
def read_file(path: str) -> str:
    """
    读取不同类型的文件并进行智能解析。
    参数: path (str): 文件路径
    返回: str: AI 对文件内容的分析结果
    """
    # 安全闸门：拦截敏感文件、超大文件、不存在或非普通文件的路径，避免被 prompt
    # 注入诱导去读 SSH/凭据/历史数据库或喂入畸形大文件拖垮进程。
    error = _validate_read_path(path)
    if error:
        return error
    # 先用 markitdown 把任何格式（pdf/docx/xlsx/...）转成统一的 Markdown 文本
    data = markitdown(path)
    # 再交给 LLM 做语义分析；结构化文件说字段，代码文件解释逻辑
    resp = model.invoke(f"分析这个文件,解释内容，"
                 f"包括结构、用途和关键信息。"
                 f"如果是代码就解释逻辑，如果是数据文件就说明字段，如果是二进制文件就说明用途。"
                 f"以下是文件:\n{data}")
    return resp.content

def markitdown(file_path: str) -> str:
    """把任意支持的文件类型转成 Markdown 文本，作为 read_file 的预处理步骤。"""
    from markitdown import MarkItDown  # 重依赖：仅在真正读文件时导入

    md = MarkItDown(enable_plugins=True)
    result = md.convert(file_path)
    return result.text_content


# read_file 安全限制：默认 10 MB，可通过环境变量 ASKANSWER_READ_FILE_MAX_BYTES 调整。
# 调用期读取 env；模块级变量非 None 时优先（测试 / 宿主硬覆盖用）。
_READ_FILE_MAX_BYTES: int | None = None
_DEFAULT_READ_FILE_MAX_BYTES = 10 * 1024 * 1024


def read_file_max_bytes() -> int:
    if _READ_FILE_MAX_BYTES is not None:
        return _READ_FILE_MAX_BYTES
    return _env_int("ASKANSWER_READ_FILE_MAX_BYTES", _DEFAULT_READ_FILE_MAX_BYTES)

# 命中即拒绝读取的敏感文件名/扩展（不区分大小写）。
# 设计：宁可误拒少数同名普通文件，也不能让 LLM 把凭据/密钥读进上下文。
_SENSITIVE_PATH_RE = re.compile(
    r"""(?ix)
    (?:^|/)(?:
        \.env(?:\.[^/]*)?$            # .env / .env.local / .env.production
      | id_(?:rsa|dsa|ed25519|ecdsa)  # SSH 私钥
      | authorized_keys$
      | known_hosts$
      | credentials$                  # AWS / gcloud
      | \.pgpass$
      | \.netrc$
      | \.htpasswd$
      | shadow$
      | wallet\.dat$
    )
    | \.(?:pem|key|p12|pfx|kdbx|gpg|asc|jks|keystore)$
    """,
)


def _validate_read_path(path: str) -> str | None:
    """读文件前的安全校验。返回错误字符串表示拒绝；None 表示放行。"""
    raw = (path or "").strip()
    if not raw:
        return "未提供文件路径"
    try:
        # 只解析符号链接到真实路径用于敏感检测；下游仍用原始路径让 markitdown 处理
        resolved = Path(raw).expanduser().resolve()
    except (OSError, ValueError) as exc:
        return f"路径解析失败：{exc}"
    if not resolved.exists():
        return f"文件不存在：{raw}"
    if not resolved.is_file():
        return f"路径不是普通文件：{raw}"
    # 同时检查原始路径与解析后路径，避免符号链接绕过
    for candidate in {raw, str(resolved)}:
        if _SENSITIVE_PATH_RE.search(candidate):
            return f"已拒绝读取敏感文件：{Path(candidate).name}"
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        return f"获取文件大小失败：{exc}"
    limit = read_file_max_bytes()
    if size > limit:
        return (
            f"文件过大（{size} 字节 > 上限 {limit}）；"
            f"如需读取，调整 ASKANSWER_READ_FILE_MAX_BYTES 环境变量。"
        )
    return None


# write_file 安全限制：默认 1 MB，可通过环境变量 ASKANSWER_WRITE_FILE_MAX_BYTES 调整。
# 写入内容会随规划一起进入 checkpoint（tool_call args），上限刻意比读取小一个量级。
_WRITE_FILE_MAX_BYTES: int | None = None
_DEFAULT_WRITE_FILE_MAX_BYTES = 1024 * 1024
# fs_write 确认框里 diff / 预览的最大行数，超出截断
_WRITE_DIFF_MAX_LINES = 200
_WRITE_PREVIEW_MAX_LINES = 40


def write_file_max_bytes() -> int:
    if _WRITE_FILE_MAX_BYTES is not None:
        return _WRITE_FILE_MAX_BYTES
    return _env_int("ASKANSWER_WRITE_FILE_MAX_BYTES", _DEFAULT_WRITE_FILE_MAX_BYTES)


def validate_write_path(path: str, content: str) -> str | None:
    """写文件前的安全校验。返回错误字符串表示拒绝；None 表示放行。

    与 ``_validate_read_path`` 同源的敏感路径正则 —— 能骗 LLM 读凭据的路径，
    同样不允许被写入（覆盖 authorized_keys / .env 比读取更危险）。
    在“规划确认内容”与“用户批准后执行”两个时机都会被调用（对齐 shell 的双重检查）。
    """
    raw = (path or "").strip()
    if not raw:
        return "未提供文件路径"
    try:
        resolved = Path(raw).expanduser().resolve()
    except (OSError, ValueError) as exc:
        return f"路径解析失败：{exc}"
    # 同时检查原始路径与解析后路径，避免符号链接绕过
    for candidate in {raw, str(resolved)}:
        if _SENSITIVE_PATH_RE.search(candidate):
            return f"已拒绝写入敏感文件：{Path(candidate).name}"
    if resolved.exists() and not resolved.is_file():
        return f"路径已存在且不是普通文件：{raw}"
    parent = resolved.parent
    if not parent.is_dir():
        # 刻意不自动创建目录：目录级副作用应该由用户显式完成
        return f"父目录不存在：{parent}"
    size = len(content.encode("utf-8", errors="replace"))
    limit = write_file_max_bytes()
    if size > limit:
        return (
            f"写入内容过大（{size} 字节 > 上限 {limit}）；"
            f"如需写入，调整 ASKANSWER_WRITE_FILE_MAX_BYTES 环境变量。"
        )
    return None


def describe_write(path: str, content: str) -> dict:
    """生成 fs_write 确认框所需的展示信息：是否覆盖、大小、diff 或内容预览。

    只返回“派生信息”—— 完整 content 已随 tool_call args 存在 checkpoint 的消息
    历史里，这里不再复制一份，避免 pending_confirmations 膨胀。
    """
    target = Path((path or "").strip()).expanduser().resolve()
    exists = target.is_file()
    try:
        size_before = target.stat().st_size if exists else 0
    except OSError:
        size_before = 0
    diff = ""
    if exists:
        try:
            old_text = target.read_text(encoding="utf-8", errors="replace")
            diff_lines = list(
                difflib.unified_diff(
                    old_text.splitlines(),
                    content.splitlines(),
                    fromfile=f"{path}（当前）",
                    tofile=f"{path}（写入后）",
                    lineterm="",
                )
            )
            if len(diff_lines) > _WRITE_DIFF_MAX_LINES:
                omitted = len(diff_lines) - _WRITE_DIFF_MAX_LINES
                diff_lines = diff_lines[:_WRITE_DIFF_MAX_LINES] + [
                    f"…（diff 已截断，省略 {omitted} 行）"
                ]
            diff = "\n".join(diff_lines) if diff_lines else "（内容无变化）"
        except Exception:
            # 旧文件读不出来（如二进制）就退化为内容预览
            diff = ""
    preview = ""
    if not diff:
        lines = content.splitlines()
        preview = "\n".join(lines[:_WRITE_PREVIEW_MAX_LINES])
        if len(lines) > _WRITE_PREVIEW_MAX_LINES:
            preview += f"\n…（预览已截断，共 {len(lines)} 行）"
    return {
        "exists": exists,
        "size_before": size_before,
        "size_after": len(content.encode("utf-8", errors="replace")),
        "diff": diff,
        "preview": preview,
    }


@tool
def write_file(path: str, content: str) -> str:
    """
    把文本内容写入指定文件（新建或覆盖已有文件）。
    敏感路径（.env、SSH 密钥、凭据等）会被自动拦截；
    写入前会暂停图流程，向用户展示 diff / 内容预览，用户确认后才真正落盘。
    参数：path 目标文件路径；content 要写入的完整文本内容。
    """
    # 正常情况下，此工具的 tool_call 会被 tools_node 拦截并走 fs_write 人机确认流程；
    # 这里仅作为“脱离图流程被直接调用”时的兜底，不落盘、要求使用图入口。
    error = validate_write_path(path, content)
    if error:
        return error
    return (
        "此工具需在图流程中运行以获得用户确认。\n"
        f"拟写入：{path}（{len(content.encode('utf-8', errors='replace'))} 字节）"
    )
