"""惰性 bootstrap：import 不改 os.environ、不建 state.db；首次使用时才加载；可 opt-out。

"导入无副作用"必须在**干净子进程**里验证（同进程早已 import 过）。其余触发点在本进程内
验证：``fresh_bootstrap`` fixture 切到临时项目 / HOME、清掉相关 env、重置 bootstrap
标记，用例结束后完整还原 ``os.environ`` 并恢复测试会话的 opt-out。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path):
    """一个带 .env 与 .askanswer/settings.json 的临时项目 + 独立 HOME。"""
    root = tmp_path / "proj"
    (root / ".askanswer").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / ".env").write_text("FROM_DOTENV=dotenv\nSHARED=dotenv\n", encoding="utf-8")
    (root / ".askanswer" / "settings.json").write_text(
        json.dumps(
            {
                "model": "anthropic:claude-test",
                "db_dsn": "sqlite:///from-settings.db",
                "env": {"FROM_SETTINGS": "settings", "SHARED": "settings"},
            }
        ),
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir()
    return root, home


def _run(project, code: str) -> dict:
    root, home = project
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    script = textwrap.dedent(code) + "\nimport json as _j; print(_j.dumps(OUT))\n"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_import_has_no_side_effects(project):
    out = _run(project, """
        import os
        before = dict(os.environ)
        import askanswer
        from askanswer.config import settings
        OUT = {
            "env_unchanged": dict(os.environ) == before,
            "bootstrapped": settings.is_bootstrapped(),
            "home_db": os.path.exists(os.path.expanduser("~/.askanswer/state.db")),
            "exports": sorted(askanswer.__all__),
        }
    """)
    assert out["env_unchanged"] is True
    assert out["bootstrapped"] is False
    assert out["home_db"] is False
    assert out["exports"] == sorted([
        "ContextSchema", "RunEvent", "create_search_assistant", "normalize_context",
        "query_input", "resume_input", "run_leg", "stream_leg", "thread_config",
    ])


_TOUCHED = (
    "FROM_DOTENV", "FROM_SETTINGS", "SHARED", "ASKANSWER_DB_DSN", "ASKANSWER_DEFAULT_MODEL",
    "ASKANSWER_DB_PATH", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "ASKANSWER_SETTINGS",
)


@pytest.fixture
def fresh_bootstrap(project, monkeypatch):
    """进程内模拟"尚未 bootstrap"：临时 cwd/HOME、干净 env、重置标记与默认模型标签。"""
    from askanswer.config import settings
    from askanswer.llm.proxy import model

    root, home = project
    saved_env = dict(os.environ)
    saved_model = (model._inner, model._spec, model._provider)
    monkeypatch.chdir(root)
    monkeypatch.setenv("HOME", str(home))
    for key in _TOUCHED:
        monkeypatch.delenv(key, raising=False)
    model._swap(None, None, None)
    settings.reset_bootstrap()
    yield root
    settings.bootstrap_environ(enabled=False)  # 恢复测试会话的 opt-out
    model._swap(*saved_model)
    os.environ.clear()
    os.environ.update(saved_env)


def test_create_search_assistant_bootstraps_lazily(fresh_bootstrap, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver

    from askanswer import create_search_assistant
    from askanswer.config import settings

    monkeypatch.setenv("SHARED", "process")  # 进程已有 env 优先级最高
    assert settings.is_bootstrapped() is False
    assert "FROM_DOTENV" not in os.environ
    create_search_assistant(InMemorySaver())
    assert settings.is_bootstrapped() is True
    assert os.environ["FROM_DOTENV"] == "dotenv"
    assert os.environ["FROM_SETTINGS"] == "settings"
    assert os.environ["SHARED"] == "process"


def test_runtime_context_from_env_bootstraps(fresh_bootstrap):
    from askanswer.core.runner import runtime_context_from_env

    assert runtime_context_from_env().db_dsn == "sqlite:///from-settings.db"


def test_default_model_label_resolved_after_bootstrap(fresh_bootstrap):
    from askanswer.llm.proxy import current_model_label, model

    assert repr(model) == "<ModelProxy (default, unresolved)>"
    assert current_model_label() == "anthropic:claude-test"


def test_model_for_bootstraps_before_reading_routes(fresh_bootstrap):
    from askanswer.config import settings
    from askanswer.llm import routing

    routing.reset_router()
    try:
        routing.model_for(routing.ROLE_CLASSIFY)
        assert settings.is_bootstrapped() is True
    finally:
        routing.reset_router()


def test_get_persistence_bootstraps_db_path(fresh_bootstrap, tmp_path):
    from askanswer.storage import get_persistence, shutdown_persistence

    settings_file = fresh_bootstrap / ".askanswer" / "settings.json"
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    data["db_path"] = str(tmp_path / "custom" / "state.db")
    settings_file.write_text(json.dumps(data), encoding="utf-8")
    try:
        assert get_persistence().db_path == tmp_path / "custom" / "state.db"
    finally:
        shutdown_persistence()
    assert (tmp_path / "custom" / "state.db").exists()


def test_opt_out_skips_file_config(fresh_bootstrap):
    from langgraph.checkpoint.memory import InMemorySaver

    from askanswer import create_search_assistant
    from askanswer.config.settings import bootstrap_environ, ensure_bootstrapped
    from askanswer.llm.proxy import current_model_label

    bootstrap_environ(enabled=False)
    create_search_assistant(InMemorySaver())
    ensure_bootstrapped()
    assert os.environ.get("FROM_DOTENV") is None
    assert os.environ.get("FROM_SETTINGS") is None
    assert current_model_label() == "openai:gpt-5.4"


def test_explicit_bootstrap_marks_process(fresh_bootstrap):
    from askanswer.config import settings

    settings.bootstrap_environ()
    assert settings.is_bootstrapped() is True
    assert os.environ["FROM_SETTINGS"] == "settings"


def test_bootstrap_into_custom_dict_does_not_mark_process(tmp_path):
    from askanswer.config import settings

    was = settings.is_bootstrapped()
    target: dict[str, str] = {}
    settings.bootstrap_environ(paths=[], environ=target, dotenv_path=tmp_path / "none.env")
    assert settings.is_bootstrapped() is was
