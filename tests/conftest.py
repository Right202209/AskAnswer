"""共享 fixture：所有测试在隔离的 env / 临时 SQLite 下运行，不依赖 API key。"""

from __future__ import annotations

import pytest

# 会影响被测行为的环境变量：每个测试开始前全部清掉，保证结果与开发机 .env 无关。
_ISOLATED_ENV_VARS = (
    "ASKANSWER_TENANT_ID",
    "ASKANSWER_DB_PATH",
    "ASKANSWER_MCP_PROFILE",
    "ASKANSWER_OTEL_EXPORTER",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "WLANGGRAPH_POSTGRES_DSN",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    for var in _ISOLATED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def pm(tmp_path):
    """临时库上的 PersistenceManager；用完关闭连接，避免 Windows/WAL 句柄残留。"""
    from askanswer.persistence import PersistenceManager

    manager = PersistenceManager(tmp_path / "state.db")
    yield manager
    manager.close()
