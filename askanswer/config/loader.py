"""从环境变量构造 ``Settings``：全项目唯一读取 ASKANSWER_* 配置变量的地方。

环境变量本身已由 ``askanswer.config.layers.bootstrap_environ`` 按分层优先级装载
（进程 env > settings.local.json > settings.json > user settings > .env）。
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from askanswer.config.schema import ROLE_NAMES, Limits, ModelRoles, Secrets, Settings

_TRUTHY = {"1", "true", "yes", "on"}


def _text(env: Mapping[str, str], key: str) -> str | None:
    value = (env.get(key) or "").strip()
    return value or None


def _positive_int(env: Mapping[str, str], key: str) -> int | None:
    raw = _text(env, key)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _digest(env: Mapping[str, str]) -> str:
    raw = (_text(env, "ASKANSWER_CONTEXT_DIGEST") or "").lower()
    if raw in ("1", "true", "yes", "on", "brief"):
        return "brief"
    if raw == "llm":
        return "llm"
    return "off"  # 未识别值一律回退 off（非回归）


def _csv(raw: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (raw or "").split(",") if item.strip())


def _models(env: Mapping[str, str]) -> ModelRoles:
    specs = {role: _text(env, f"ASKANSWER_MODEL_{role.upper()}") for role in ROLE_NAMES}
    fallbacks = {}
    for role in ROLE_NAMES:
        chain = _csv(env.get(f"ASKANSWER_MODEL_FALLBACKS_{role.upper()}"))
        if chain:
            fallbacks[role] = chain
    return ModelRoles(**specs, fallbacks=fallbacks)


def _limits(env: Mapping[str, str]) -> Limits:
    values = {}
    for field, key in (
        ("read_file_max_bytes", "ASKANSWER_READ_FILE_MAX_BYTES"),
        ("read_file_max_chars", "ASKANSWER_READ_FILE_MAX_CHARS"),
        ("write_file_max_bytes", "ASKANSWER_WRITE_FILE_MAX_BYTES"),
        ("shell_output_max_bytes", "ASKANSWER_SHELL_OUTPUT_MAX_BYTES"),
    ):
        value = _positive_int(env, key)
        if value is not None:
            values[field] = value
    return Limits(**values)


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """按当前（或给定的）环境变量构造一份不可变 ``Settings``。

    非法值不抛异常、回退默认（与历史行为一致：配置写错不应让 CLI 起不来）。
    """
    env = os.environ if environ is None else environ
    return Settings(
        default_model=_text(env, "ASKANSWER_DEFAULT_MODEL") or "openai:gpt-5.4",
        models=_models(env),
        tenant_id=_text(env, "ASKANSWER_TENANT_ID"),
        db_path=_text(env, "ASKANSWER_DB_PATH"),
        db_dsn=_text(env, "WLANGGRAPH_POSTGRES_DSN"),
        db_dialect=_text(env, "ASKANSWER_DB_DIALECT"),
        mcp_profile=_text(env, "ASKANSWER_MCP_PROFILE"),
        mcp_all_intents=(_text(env, "ASKANSWER_MCP_ALL_INTENTS") or "").lower() in _TRUTHY,
        deny_tools=frozenset(_csv(env.get("ASKANSWER_DENY_TOOLS"))),
        context_max_tokens=_positive_int(env, "ASKANSWER_CONTEXT_MAX_TOKENS"),
        context_digest=_digest(env),
        run_token_budget=_positive_int(env, "ASKANSWER_RUN_TOKEN_BUDGET"),
        data_home=_text(env, "XDG_DATA_HOME"),
        otel_exporter=_text(env, "ASKANSWER_OTEL_EXPORTER"),
        limits=_limits(env),
        secrets=Secrets(
            tavily_api_key=_text(env, "TAVILY_API_KEY"),
            openweather_api_key=_text(env, "OPENWEATHER_API_KEY"),
            server_token=_text(env, "ASKANSWER_SERVER_TOKEN"),
        ),
    )


def current_settings() -> Settings:
    """当前进程配置的快照（每次调用重新构造，成本是微秒级）。

    刻意不缓存：``/model``、settings 热改、单测 monkeypatch.setenv 都能立即生效。
    需要固定一份配置时，由组装根（``runtime.container``）持有并注入。
    """
    return load_settings()
