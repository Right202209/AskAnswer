"""分层 settings.json 配置（类似 Claude Code 的 settings.json）。

与 ``.env`` 并存：本模块读 JSON 并注入环境变量；``load.py`` 按完整优先级
调用 ``load_dotenv`` + ``apply_settings``（见 ``bootstrap_environ``）。

**全局优先级**（高 → 低）：

1. **进程已有环境变量**（shell export / CI / 父进程）— 永不被文件覆盖
2. **Local**  — ``<project>/.askanswer/settings.local.json``（个人、勿提交）
3. **Project** — ``<project>/.askanswer/settings.json``（可提交团队共享）
4. **User**    — ``~/.askanswer/settings.json``（跨项目个人默认）
5. **``.env``** — 最低保底（``load_dotenv(override=False)``）

文件格式（与 Claude Code 的 ``env`` 块同构，并支持若干一等字段展开为 env）：

.. code-block:: json

    {
      "env": {
        "OPENAI_API_KEY": "...",
        "ASKANSWER_MODEL_CLASSIFY": "openai:gpt-4o-mini"
      },
      "model": "openai:gpt-5.4",
      "models": {
        "classify": "openai:gpt-4o-mini",
        "evaluate": "openai:gpt-4o-mini",
        "summarize": "openai:gpt-4o-mini",
        "answer": "openai:gpt-5.4",
        "fallbacks": {
          "answer": ["openai:gpt-4o", "deepseek:deepseek-chat"]
        }
      },
      "tenant_id": "alice",
      "db_path": "~/.askanswer/state.db",
      "context": {
        "max_tokens": 24000,
        "digest": "brief"
      },
      "run_token_budget": 60000,
      "mcp_all_intents": true
    }

损坏 / 缺失的文件被静默忽略，不阻塞启动（与 ``mcp_profile`` 一致）。
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

# ── paths ────────────────────────────────────────────────────────────

_USER_DIR_NAME = ".askanswer"
_PROJECT_DIR_NAME = ".askanswer"
_SETTINGS_FILENAME = "settings.json"
_SETTINGS_LOCAL_FILENAME = "settings.local.json"

# 向上找项目根时的标记文件/目录（命中任一即视为项目根）
_PROJECT_MARKERS = (
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    ".askanswer",
)


def user_settings_path() -> Path:
    """``~/.askanswer/settings.json``（可被 ``XDG_CONFIG_HOME`` / ``ASKANSWER_SETTINGS`` 覆盖）。"""
    explicit = os.environ.get("ASKANSWER_SETTINGS")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg).expanduser() / "askanswer" / _SETTINGS_FILENAME
    return Path.home() / _USER_DIR_NAME / _SETTINGS_FILENAME


def find_project_root(start: Path | None = None) -> Path:
    """从 ``start``（默认 cwd）向上查找项目根；找不到则返回 ``start`` 本身。"""
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        for marker in _PROJECT_MARKERS:
            if (candidate / marker).exists():
                return candidate
    return cur


def project_settings_path(root: Path | None = None) -> Path:
    base = root if root is not None else find_project_root()
    return base / _PROJECT_DIR_NAME / _SETTINGS_FILENAME


def local_settings_path(root: Path | None = None) -> Path:
    base = root if root is not None else find_project_root()
    return base / _PROJECT_DIR_NAME / _SETTINGS_LOCAL_FILENAME


def settings_paths(*, cwd: Path | None = None) -> list[Path]:
    """按**从低到高**优先级返回将要合并的 settings 文件路径。

    顺序：user → project → local。后面的覆盖前面的。
    """
    root = find_project_root(cwd)
    return [
        user_settings_path(),
        project_settings_path(root),
        local_settings_path(root),
    ]


# ── load / merge ─────────────────────────────────────────────────────

def load_json_file(path: Path) -> dict[str, Any]:
    """读取单个 settings 文件；缺失或损坏时返回 ``{}``。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """浅+递归合并：两边都是 dict 的 key 递归合并，其余 key 直接覆盖。

    ``env`` 等扁平映射走覆盖；嵌套对象（如 ``models`` / ``context``）递归。
    """
    out: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[key] = deep_merge(existing, value)
        else:
            out[key] = value
    return out


def load_merged(*, cwd: Path | None = None, paths: Iterable[Path] | None = None) -> dict[str, Any]:
    """按优先级合并所有 settings 文件，返回一份扁平化前的配置 dict。"""
    merged: dict[str, Any] = {}
    for path in (paths if paths is not None else settings_paths(cwd=cwd)):
        piece = load_json_file(Path(path))
        if piece:
            merged = deep_merge(merged, piece)
    return merged


# ── expand first-class keys → env ────────────────────────────────────

def _stringify(value: Any) -> str | None:
    """把 JSON 值变成可写入 ``os.environ`` 的字符串；无法表示则返回 None。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (str, int, float)):
        return str(value)
    return None


def expand_to_env(settings: Mapping[str, Any]) -> dict[str, str]:
    """把 settings 展开成 env 键值对。

    规则：
    - ``env`` 字典原样展开（值必须可 stringify）
    - 一等字段映射到 ASKANSWER_* / 常见密钥名（见下）
    - 同名时：一等字段先写，``env`` 块后写 → ``env`` 显式覆盖一等字段
    """
    env: dict[str, str] = {}

    # ── first-class convenience keys ──
    model = settings.get("model")
    if isinstance(model, str) and model.strip():
        # 默认全局模型标签；load.py 读取 ASKANSWER_DEFAULT_MODEL
        env["ASKANSWER_DEFAULT_MODEL"] = model.strip()

    models = settings.get("models")
    if isinstance(models, dict):
        for role in ("classify", "evaluate", "summarize", "answer"):
            spec = models.get(role)
            if isinstance(spec, str) and spec.strip():
                env[f"ASKANSWER_MODEL_{role.upper()}"] = spec.strip()
        fallbacks = models.get("fallbacks")
        if isinstance(fallbacks, dict):
            for role, chain in fallbacks.items():
                if not isinstance(role, str):
                    continue
                if isinstance(chain, list):
                    parts = [str(x).strip() for x in chain if str(x).strip()]
                    if parts:
                        env[f"ASKANSWER_MODEL_FALLBACKS_{role.upper()}"] = ",".join(parts)
                elif isinstance(chain, str) and chain.strip():
                    env[f"ASKANSWER_MODEL_FALLBACKS_{role.upper()}"] = chain.strip()

    for key, env_name in (
        ("tenant_id", "ASKANSWER_TENANT_ID"),
        ("tenantId", "ASKANSWER_TENANT_ID"),
        ("db_path", "ASKANSWER_DB_PATH"),
        ("dbPath", "ASKANSWER_DB_PATH"),
        ("mcp_profile", "ASKANSWER_MCP_PROFILE"),
        ("mcpProfile", "ASKANSWER_MCP_PROFILE"),
        ("mcp_all_intents", "ASKANSWER_MCP_ALL_INTENTS"),
        ("mcpAllIntents", "ASKANSWER_MCP_ALL_INTENTS"),
        ("run_token_budget", "ASKANSWER_RUN_TOKEN_BUDGET"),
        ("runTokenBudget", "ASKANSWER_RUN_TOKEN_BUDGET"),
        ("db_dialect", "ASKANSWER_DB_DIALECT"),
        ("dbDialect", "ASKANSWER_DB_DIALECT"),
        ("server_token", "ASKANSWER_SERVER_TOKEN"),
        ("serverToken", "ASKANSWER_SERVER_TOKEN"),
    ):
        if key in settings:
            text = _stringify(settings[key])
            if text is not None:
                env[env_name] = text

    context = settings.get("context")
    if isinstance(context, dict):
        if "max_tokens" in context or "maxTokens" in context:
            raw = context.get("max_tokens", context.get("maxTokens"))
            text = _stringify(raw)
            if text is not None:
                env["ASKANSWER_CONTEXT_MAX_TOKENS"] = text
        if "digest" in context:
            text = _stringify(context["digest"])
            if text is not None:
                env["ASKANSWER_CONTEXT_DIGEST"] = text

    # ── explicit env block last (wins over first-class for same key) ──
    raw_env = settings.get("env")
    if isinstance(raw_env, dict):
        for key, value in raw_env.items():
            if not isinstance(key, str) or not key:
                continue
            text = _stringify(value)
            if text is not None:
                env[key] = text

    return env


# ── apply ────────────────────────────────────────────────────────────

def apply_settings(
    *,
    cwd: Path | None = None,
    paths: Iterable[Path] | None = None,
    environ: dict[str, str] | None = None,
    override: bool = False,
    protect: Collection[str] | None = None,
) -> dict[str, str]:
    """加载并合并 settings，把展开后的 env 写入 ``environ``（默认 ``os.environ``）。

    Parameters
    ----------
    override:
        ``False``（默认）时用 setdefault：不覆盖 ``target`` 里已有的变量。
        ``True`` 时强制写入（用于盖过 ``.env`` 保底值）；仍尊重 ``protect``。
    protect:
        这些键**绝不会**被写入（即使 ``override=True``）。``bootstrap_environ``
        传入启动前已存在的进程 env 键，保证 shell/CI 变量优先级最高。

    Returns
    -------
    settings 展开得到的 env 字典副本（不论是否实际写入），便于诊断与单测。
    """
    target: dict[str, str] = environ if environ is not None else os.environ  # type: ignore[assignment]
    protected = set(protect) if protect is not None else set()
    merged = load_merged(cwd=cwd, paths=paths)
    if not merged:
        return {}
    env = expand_to_env(merged)
    for key, value in env.items():
        if key in protected:
            continue
        if override or key not in target:
            target[key] = value
    return dict(env)


def bootstrap_environ(
    *,
    cwd: Path | None = None,
    paths: Iterable[Path] | None = None,
    dotenv_path: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """按正式优先级装载配置：``.env`` 保底 → settings 覆盖 → 保护进程已有 env。

    1. 快照当前 ``environ`` 中已有的键（进程 / shell / CI）
    2. ``load_dotenv(override=False)`` — 只填空缺，``.env`` 优先级最低
    3. ``apply_settings(override=True, protect=快照)`` — settings 盖过 ``.env``，
       但不碰步骤 1 里已存在的键

    Returns
    -------
    settings 展开的 env 字典（同 ``apply_settings``）。
    """
    from dotenv import load_dotenv

    target: dict[str, str] = environ if environ is not None else os.environ  # type: ignore[assignment]
    preexisting = frozenset(target)
    # python-dotenv 默认写 os.environ；传入 dict 时需走 stream 接口。
    if environ is None:
        load_dotenv(dotenv_path, override=False)
    else:
        _load_dotenv_into(target, dotenv_path, override=False)
    return apply_settings(
        cwd=cwd,
        paths=paths,
        environ=target,
        override=True,
        protect=preexisting,
    )


def _load_dotenv_into(
    target: dict[str, str],
    dotenv_path: str | Path | None,
    *,
    override: bool,
) -> None:
    """把 ``.env`` 解析结果写入任意 dict（便于单测隔离，不污染真实 os.environ）。"""
    from dotenv import dotenv_values

    path = dotenv_path
    if path is None:
        # 与 load_dotenv() 默认行为一致：从 cwd 向上找 .env
        path = Path.cwd() / ".env"
        if not path.is_file():
            return
    values = dotenv_values(path)
    for key, value in values.items():
        if key is None or value is None:
            continue
        if override or key not in target:
            target[key] = value


def describe_sources(*, cwd: Path | None = None) -> list[tuple[str, Path, bool]]:
    """返回 ``(scope, path, exists)`` 列表，供 ``/status`` 或诊断使用。"""
    root = find_project_root(cwd)
    return [
        ("user", user_settings_path(), user_settings_path().is_file()),
        ("project", project_settings_path(root), project_settings_path(root).is_file()),
        ("local", local_settings_path(root), local_settings_path(root).is_file()),
    ]
