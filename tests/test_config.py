"""P3：类型化 Settings —— 环境变量只在 config/loader.py 里读取。"""

from __future__ import annotations

from pathlib import Path

from askanswer.config import load_settings
from askanswer.config.paths import db_path, history_dir, mcp_profile_path


def test_defaults_without_env():
    s = load_settings({})
    assert s.default_model == "openai:gpt-5.4"
    assert s.tenant_id is None
    assert s.context_digest == "off"
    assert s.run_token_budget is None
    assert s.limits.write_file_max_bytes == 1024 * 1024
    assert s.deny_tools == frozenset()


def test_parses_typed_fields():
    s = load_settings({
        "ASKANSWER_DEFAULT_MODEL": "deepseek:deepseek-chat",
        "ASKANSWER_MODEL_CLASSIFY": "openai:gpt-4o-mini",
        "ASKANSWER_MODEL_FALLBACKS_ANSWER": "openai:gpt-4o, deepseek:deepseek-chat",
        "ASKANSWER_TENANT_ID": "alice",
        "ASKANSWER_MCP_ALL_INTENTS": "yes",
        "ASKANSWER_DENY_TOOLS": "pwd, tavily_search",
        "ASKANSWER_CONTEXT_MAX_TOKENS": "24000",
        "ASKANSWER_CONTEXT_DIGEST": "true",
        "ASKANSWER_RUN_TOKEN_BUDGET": "60000",
        "ASKANSWER_WRITE_FILE_MAX_BYTES": "10",
        "TAVILY_API_KEY": "tvly-secret",
    })
    assert s.default_model == "deepseek:deepseek-chat"
    assert s.models.spec("classify") == "openai:gpt-4o-mini"
    assert s.models.spec("answer") is None
    assert s.models.fallback_chain("answer") == ("openai:gpt-4o", "deepseek:deepseek-chat")
    assert s.tenant_id == "alice"
    assert s.mcp_all_intents is True
    assert s.deny_tools == {"pwd", "tavily_search"}
    assert s.context_max_tokens == 24000
    assert s.context_digest == "brief"
    assert s.run_token_budget == 60000
    assert s.limits.write_file_max_bytes == 10
    assert "tvly-secret" not in repr(s)  # SecretStr 不在 repr 中泄漏


def test_invalid_values_fall_back_to_defaults():
    s = load_settings({
        "ASKANSWER_RUN_TOKEN_BUDGET": "abc",
        "ASKANSWER_CONTEXT_MAX_TOKENS": "-5",
        "ASKANSWER_CONTEXT_DIGEST": "weird",
        "ASKANSWER_READ_FILE_MAX_BYTES": "0",
    })
    assert s.run_token_budget is None
    assert s.context_max_tokens is None
    assert s.context_digest == "off"
    assert s.limits.read_file_max_bytes == 10 * 1024 * 1024


def test_paths_follow_one_convention(tmp_path):
    s = load_settings({"XDG_DATA_HOME": str(tmp_path)})
    assert db_path(s) == tmp_path / "askanswer" / "state.db"
    assert history_dir(s) == tmp_path / "askanswer"
    assert mcp_profile_path(s) == tmp_path / "askanswer" / "mcp.json"
    explicit = load_settings({"ASKANSWER_DB_PATH": str(tmp_path / "x" / "s.db")})
    assert db_path(explicit) == tmp_path / "x" / "s.db"
    assert history_dir(explicit) == tmp_path / "x"


def test_no_module_reads_askanswer_env_directly():
    """除 config/loader.py 与 settings.py（分层装载）外，业务代码不直接读配置变量。"""
    root = Path(__file__).resolve().parents[1] / "askanswer"
    allowed = {root / "config" / "loader.py", root / "config" / "layers.py"}
    offenders = []
    for path in root.rglob("*.py"):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in ('environ.get("ASKANSWER_', 'getenv("ASKANSWER_'):
            if needle in text:
                offenders.append(str(path.relative_to(root)))
    assert offenders == []
