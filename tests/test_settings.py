"""settings.json 分层加载：合并、env 展开、setdefault 语义、损坏容错。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from askanswer import settings


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """带 .git 标记的假项目根，避免 walk-up 误撞到真实仓库。"""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".askanswer").mkdir()
    return tmp_path


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_missing_returns_empty(tmp_path: Path):
    assert settings.load_json_file(tmp_path / "nope.json") == {}


def test_corrupt_json_tolerated(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert settings.load_json_file(p) == {}


def test_non_object_root_tolerated(tmp_path: Path):
    p = tmp_path / "arr.json"
    p.write_text("[1, 2]", encoding="utf-8")
    assert settings.load_json_file(p) == {}


def test_deep_merge_overlays_scalars_and_dicts():
    base = {"env": {"A": "1", "B": "2"}, "context": {"max_tokens": 1000, "digest": "brief"}}
    over = {"env": {"B": "9", "C": "3"}, "context": {"max_tokens": 2000}, "tenant_id": "t"}
    merged = settings.deep_merge(base, over)
    assert merged["env"] == {"A": "1", "B": "9", "C": "3"}
    assert merged["context"] == {"max_tokens": 2000, "digest": "brief"}
    assert merged["tenant_id"] == "t"


def test_load_merged_priority_user_project_local(project_root: Path, monkeypatch, tmp_path: Path):
    user = tmp_path / "user-settings.json"
    _write(user, {"env": {"K": "user", "U": "1"}, "tenant_id": "from-user"})
    _write(
        project_root / ".askanswer" / "settings.json",
        {"env": {"K": "project", "P": "1"}, "tenant_id": "from-project"},
    )
    _write(
        project_root / ".askanswer" / "settings.local.json",
        {"env": {"K": "local"}, "tenant_id": "from-local"},
    )
    monkeypatch.setenv("ASKANSWER_SETTINGS", str(user))

    merged = settings.load_merged(cwd=project_root)
    assert merged["env"]["K"] == "local"
    assert merged["env"]["U"] == "1"
    assert merged["env"]["P"] == "1"
    assert merged["tenant_id"] == "from-local"


def test_expand_env_block_and_first_class_keys():
    data = {
        "model": "anthropic:claude-sonnet-4-5",
        "models": {
            "classify": "openai:gpt-4o-mini",
            "fallbacks": {"answer": ["openai:gpt-4o", "deepseek:deepseek-chat"]},
        },
        "tenant_id": "alice",
        "db_path": "~/x.db",
        "mcp_all_intents": True,
        "context": {"max_tokens": 24000, "digest": "llm"},
        "run_token_budget": 60000,
        "env": {
            "OPENAI_API_KEY": "sk-test",
            "ASKANSWER_TENANT_ID": "from-env-block",  # 覆盖 first-class
        },
    }
    env = settings.expand_to_env(data)
    assert env["ASKANSWER_DEFAULT_MODEL"] == "anthropic:claude-sonnet-4-5"
    assert env["ASKANSWER_MODEL_CLASSIFY"] == "openai:gpt-4o-mini"
    assert env["ASKANSWER_MODEL_FALLBACKS_ANSWER"] == "openai:gpt-4o,deepseek:deepseek-chat"
    assert env["ASKANSWER_TENANT_ID"] == "from-env-block"
    assert env["ASKANSWER_DB_PATH"] == "~/x.db"
    assert env["ASKANSWER_MCP_ALL_INTENTS"] == "1"
    assert env["ASKANSWER_CONTEXT_MAX_TOKENS"] == "24000"
    assert env["ASKANSWER_CONTEXT_DIGEST"] == "llm"
    assert env["ASKANSWER_RUN_TOKEN_BUDGET"] == "60000"
    assert env["OPENAI_API_KEY"] == "sk-test"


def test_apply_settings_setdefault_does_not_clobber(project_root: Path, monkeypatch, tmp_path: Path):
    user = tmp_path / "user.json"
    _write(user, {"env": {"OPENAI_API_KEY": "from-settings", "NEW_ONLY": "yes"}})
    monkeypatch.setenv("ASKANSWER_SETTINGS", str(user))
    # 清空项目侧文件干扰
    monkeypatch.chdir(project_root)

    target = {"OPENAI_API_KEY": "already-set"}
    written = settings.apply_settings(cwd=project_root, environ=target, override=False)
    assert written["OPENAI_API_KEY"] == "from-settings"
    assert target["OPENAI_API_KEY"] == "already-set"  # 未覆盖
    assert target["NEW_ONLY"] == "yes"


def test_apply_settings_override_true(project_root: Path, monkeypatch, tmp_path: Path):
    user = tmp_path / "user.json"
    _write(user, {"env": {"OPENAI_API_KEY": "from-settings"}})
    monkeypatch.setenv("ASKANSWER_SETTINGS", str(user))
    monkeypatch.chdir(project_root)

    target = {"OPENAI_API_KEY": "already-set"}
    settings.apply_settings(cwd=project_root, environ=target, override=True)
    assert target["OPENAI_API_KEY"] == "from-settings"


def test_apply_settings_protect_blocks_override(project_root: Path, monkeypatch, tmp_path: Path):
    user = tmp_path / "user.json"
    _write(user, {"env": {"OPENAI_API_KEY": "from-settings", "OTHER": "s"}})
    monkeypatch.setenv("ASKANSWER_SETTINGS", str(user))
    monkeypatch.chdir(project_root)

    target = {"OPENAI_API_KEY": "from-process"}
    settings.apply_settings(
        cwd=project_root,
        environ=target,
        override=True,
        protect={"OPENAI_API_KEY"},
    )
    assert target["OPENAI_API_KEY"] == "from-process"
    assert target["OTHER"] == "s"


def test_bootstrap_environ_dotenv_lowest(
    project_root: Path, monkeypatch, tmp_path: Path,
):
    """进程 > settings > .env（保底）。"""
    user = tmp_path / "user.json"
    _write(
        user,
        {"env": {"FROM_SETTINGS": "settings", "SHARED": "settings", "ONLY_SETTINGS": "s"}},
    )
    monkeypatch.setenv("ASKANSWER_SETTINGS", str(user))
    monkeypatch.chdir(project_root)

    dotenv_file = project_root / ".env"
    dotenv_file.write_text(
        "FROM_DOTENV=dotenv\nSHARED=dotenv\nONLY_DOTENV=d\nPROCESS_KEY=should-not-win\n",
        encoding="utf-8",
    )

    target = {"PROCESS_KEY": "process", "SHARED": "process"}
    settings.bootstrap_environ(
        cwd=project_root,
        dotenv_path=dotenv_file,
        environ=target,
    )

    assert target["PROCESS_KEY"] == "process"       # 进程最高
    assert target["SHARED"] == "process"            # 进程 > settings / .env
    assert target["FROM_SETTINGS"] == "settings"    # settings 写入
    assert target["ONLY_SETTINGS"] == "s"
    assert target["FROM_DOTENV"] == "dotenv"        # .env 保底
    assert target["ONLY_DOTENV"] == "d"


def test_bootstrap_environ_settings_override_dotenv(
    project_root: Path, monkeypatch, tmp_path: Path,
):
    user = tmp_path / "user.json"
    _write(user, {"env": {"SHARED": "settings"}})
    monkeypatch.setenv("ASKANSWER_SETTINGS", str(user))
    monkeypatch.chdir(project_root)

    dotenv_file = project_root / ".env"
    dotenv_file.write_text("SHARED=dotenv\nONLY_DOTENV=d\n", encoding="utf-8")

    target: dict[str, str] = {}
    settings.bootstrap_environ(
        cwd=project_root,
        dotenv_path=dotenv_file,
        environ=target,
    )
    assert target["SHARED"] == "settings"   # settings > .env
    assert target["ONLY_DOTENV"] == "d"     # .env 仍可补缺


def test_find_project_root_walks_up(tmp_path: Path):
    root = tmp_path / "repo"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert settings.find_project_root(nested) == root.resolve()


def test_describe_sources(project_root: Path, monkeypatch, tmp_path: Path):
    user = tmp_path / "u.json"
    _write(user, {})
    monkeypatch.setenv("ASKANSWER_SETTINGS", str(user))
    _write(project_root / ".askanswer" / "settings.json", {"env": {}})
    sources = settings.describe_sources(cwd=project_root)
    by_scope = {s: (p, ex) for s, p, ex in sources}
    assert by_scope["user"][1] is True
    assert by_scope["project"][1] is True
    assert by_scope["local"][1] is False


def test_bool_and_number_stringify():
    env = settings.expand_to_env({"mcp_all_intents": False, "run_token_budget": 0})
    assert env["ASKANSWER_MCP_ALL_INTENTS"] == "0"
    assert env["ASKANSWER_RUN_TOKEN_BUDGET"] == "0"
