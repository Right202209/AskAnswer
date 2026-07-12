"""MCP server 连接 profile 的读写（``~/.askanswer/mcp.json``）。

目的：让用户 ``/mcp add`` 过的 server 在下次启动时自动重连，不必每次手敲 URL。
职责边界刻意收窄：

- 只做文件级持久化，不 import ``mcp`` / ``registry`` / ``graph`` 任何运行时模块，
  避免 CLI 启动时因加载顺序触发副作用（对齐“mcp.py 顶层不 import graph”这条不变量）。
- 写入走“临时文件 + ``os.replace``”原子替换，避免并发/崩溃留下半截 JSON。
- 每条记录以 ``name`` 为主键；``save_entry`` 覆盖同名项，``remove_entry`` 按名删除。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# profile 文件名；与 state.db 放在同一 base 目录下（见 ``default_profile_path``）。
_PROFILE_FILENAME = "mcp.json"

# 单条 server 记录允许的字段白名单：读写两侧都按它过滤，防止脏字段混入。
_ALLOWED_KEYS = ("name", "transport", "url", "command", "args", "env", "headers")


def default_profile_path() -> Path:
    """返回 mcp.json 的默认路径。

    优先级与 ``persistence.default_db_path`` 保持一致：
    ``ASKANSWER_MCP_PROFILE`` > ``XDG_DATA_HOME/askanswer`` > ``~/.askanswer``。
    """
    explicit = os.environ.get("ASKANSWER_MCP_PROFILE")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() / "askanswer" if xdg else Path.home() / ".askanswer"
    return base / _PROFILE_FILENAME


def load(path: Path | None = None) -> list[dict]:
    """读取 profile 中的 server 记录列表；文件缺失/损坏时返回空列表（不抛异常）。"""
    target = path or default_profile_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 损坏的 profile 不应阻塞启动：当作空处理，后续写入会覆盖它。
        return []
    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, list):
        return []
    return [_clean_record(item) for item in servers if _is_valid_record(item)]


def save_entry(record: dict, *, path: Path | None = None) -> None:
    """新增或覆盖一条 server 记录（按 ``name`` 去重），原子写回文件。"""
    if not _is_valid_record(record):
        raise ValueError("MCP profile 记录缺少必填字段：name / transport")
    target = path or default_profile_path()
    cleaned = _clean_record(record)
    servers = [r for r in load(target) if r.get("name") != cleaned["name"]]
    servers.append(cleaned)
    _atomic_write(target, servers)


def remove_entry(name: str, *, path: Path | None = None) -> bool:
    """按名删除一条记录，返回是否真的删掉了某项。"""
    target = path or default_profile_path()
    servers = load(target)
    remaining = [r for r in servers if r.get("name") != name]
    if len(remaining) == len(servers):
        return False
    _atomic_write(target, remaining)
    return True


# ── helpers ───────────────────────────────────────────────────────────

def _is_valid_record(record) -> bool:
    """一条记录至少要有 name 与 transport 才算有效。"""
    if not isinstance(record, dict):
        return False
    return bool(record.get("name")) and bool(record.get("transport"))


def _clean_record(record: dict) -> dict:
    """只保留白名单字段，丢弃 None 值，避免脏数据写盘。"""
    return {
        key: record[key]
        for key in _ALLOWED_KEYS
        if key in record and record[key] is not None
    }


def _atomic_write(target: Path, servers: list[dict]) -> None:
    """把 servers 列表原子写入 profile：先写临时文件再 ``os.replace``。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"servers": servers}, ensure_ascii=False, indent=2)
    # 临时文件必须与目标同目录，才能保证 os.replace 是同一文件系统上的原子操作。
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp_name, target)
    except Exception:
        # 失败时清理临时文件，避免残留碎片。
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
