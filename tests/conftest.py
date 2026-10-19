"""共享 fixture：所有测试在隔离的 env / 临时 SQLite 下运行，不依赖 API key。"""

from __future__ import annotations

import pytest

# 会影响被测行为的环境变量：每个测试开始前全部清掉，保证结果与开发机 .env 无关。
_ISOLATED_ENV_VARS = (
    "ASKANSWER_TENANT_ID",
    "ASKANSWER_DB_PATH",
    "ASKANSWER_OTEL_EXPORTER",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "WLANGGRAPH_POSTGRES_DSN",
    "ASKANSWER_DB_DSN",
    "ASKANSWER_DB_DIALECT",
)


@pytest.fixture(autouse=True, scope="session")
def _no_file_config():
    """测试进程不加载开发机的 .env / settings.json：惰性 bootstrap 直接 opt-out。

    需要验证 bootstrap 行为的用例在子进程里跑（见 tests/config/test_bootstrap.py）。
    """
    from askanswer.config import settings

    settings.bootstrap_environ(enabled=False)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    for var in _ISOLATED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def pm(tmp_path):
    """临时库上的 PersistenceManager；用完关闭连接，避免 Windows/WAL 句柄残留。"""
    from askanswer.storage import PersistenceManager

    manager = PersistenceManager(tmp_path / "state.db")
    yield manager
    manager.close()


def _scripted_model_cls():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class _Scripted(BaseChatModel):
        """按脚本依次返回 AIMessage 的假模型；脚本耗尽后回 "ok"。记录每次收到的消息。"""

        responses: list = []
        calls: list = []

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls.append(list(messages))
            message = self.responses.pop(0) if self.responses else AIMessage(content="ok")
            if isinstance(message, str):
                message = AIMessage(content=message)
            return ChatResult(generations=[ChatGeneration(message=message)])

        def bind_tools(self, tools, **kwargs):
            return self

        @property
        def _llm_type(self) -> str:
            return "scripted"

    return _Scripted


@pytest.fixture
def fake_llm():
    """把全局模型代理的 backend 换成脚本化假模型；用例结束后还原。"""
    from askanswer.llm.proxy import model

    fake = _scripted_model_cls()(responses=[], calls=[])
    saved = (model._inner, model._spec, model._provider)
    model._swap(fake, "scripted", "fake")
    yield fake
    model._swap(*saved)
