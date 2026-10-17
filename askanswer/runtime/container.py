"""组装根：图节点需要的全部依赖在这里显式组装，再以参数注入。

节点由 ``make_xxx_node(container)`` 工厂生成，节点体内不再 import 模块级单例；
单测构造一个替身 Container 即可，不必 monkeypatch 模块全局变量。

``build_container()`` 不传参时使用进程默认实现（全局工具注册表、intent 注册表、
``routing.model_for``），CLI / HTTP 入口就是这么用的。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from askanswer.config import Settings, current_settings


def _constant(value: Any) -> Callable[[], Any]:
    return lambda: value


@dataclass(frozen=True)
class Container:
    """依赖集合。tools / intents 以 provider 形式持有：首次访问才构造，
    这样 ``--graph`` 导出、单纯建图都不会提前 seed 工具或连接 MCP。"""

    tools_provider: Callable[[], Any]            # () -> ToolRegistry
    intents_provider: Callable[[], Any]          # () -> IntentRegistry
    models: Callable[[str], Any]                 # role -> chat model（经路由与回退）
    confirmations: Callable[[str | None], Any]   # 确认类 -> handler | None
    settings_provider: Callable[[], Settings] = field(default=current_settings)
    store_provider: Callable[[], Any] = field(default=lambda: _default_store())
    bus: Any = field(default=None)               # EventBus；None 时用进程默认总线

    @property
    def tools(self):
        return self.tools_provider()

    @property
    def intents(self):
        return self.intents_provider()

    @property
    def store(self):
        """存储门面（首次访问才打开 SQLite）。"""
        return self.store_provider()

    @property
    def events(self):
        from askanswer.observability.bus import get_bus

        return self.bus if self.bus is not None else get_bus()

    @property
    def settings(self) -> Settings:
        """读取当前配置；默认每次取最新值（/model、settings 热改即时生效）。"""
        return self.settings_provider()


def _default_store():
    from askanswer.storage.store import get_store

    return get_store()


def _default_tools():
    from askanswer.tools.registry import get_registry

    return get_registry()


def _default_intents():
    from askanswer.intents import get_intent_registry

    return get_intent_registry()


def _default_models(role: str):
    from askanswer.llm.routing import model_for

    return model_for(role)


def _default_confirmations(clazz: str | None):
    from askanswer.safety.approvals import get_confirmation_handler

    return get_confirmation_handler(clazz)


def build_container(
    *,
    settings: Settings | None = None,
    tools: Any = None,
    intents: Any = None,
    models: Callable[[str], Any] | None = None,
    confirmations: Callable[[str | None], Any] | None = None,
    store: Any = None,
    bus: Any = None,
) -> Container:
    """组装一个 Container；未给出的依赖使用进程默认实现（延迟 import，导入无副作用）。"""
    return Container(
        tools_provider=_constant(tools) if tools is not None else _default_tools,
        intents_provider=_constant(intents) if intents is not None else _default_intents,
        models=models or _default_models,
        confirmations=confirmations or _default_confirmations,
        settings_provider=_constant(settings) if settings is not None else current_settings,
        store_provider=_constant(store) if store is not None else _default_store,
        bus=bus,
    )
