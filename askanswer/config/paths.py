"""数据目录约定：state.db / REPL history / mcp.json 都放在同一个 base 目录下。

优先级：显式路径 > ``XDG_DATA_HOME/askanswer`` > ``~/.askanswer``。
"""

from __future__ import annotations

from pathlib import Path

from .schema import Settings

_DIR_NAME = "askanswer"
_HOME_DIR_NAME = ".askanswer"


def data_dir(settings: Settings) -> Path:
    if settings.data_home:
        return Path(settings.data_home).expanduser() / _DIR_NAME
    return Path.home() / _HOME_DIR_NAME


def db_path(settings: Settings) -> Path:
    if settings.db_path:
        return Path(settings.db_path).expanduser()
    return data_dir(settings) / "state.db"


def history_dir(settings: Settings) -> Path:
    """REPL 历史与 state.db 同级（显式 db_path 时取其父目录）。"""
    return db_path(settings).parent


def mcp_profile_path(settings: Settings, filename: str = "mcp.json") -> Path:
    if settings.mcp_profile:
        return Path(settings.mcp_profile).expanduser()
    return data_dir(settings) / filename
