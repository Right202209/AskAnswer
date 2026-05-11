# 全局模型/外部客户端加载与热替换。
# - 通过 _ModelProxy 让所有模块只在启动时导入一次 model，
#   /model 切换时只替换 proxy 内部的 backend，所有引用自动指向新模型。
# - 同时初始化 Tavily 搜索客户端、读取 OpenWeather Key 等运行时常量。
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAI
from tavily import TavilyClient

from .audit import with_llm_audit_callback

# override=True：让 .env 中的值覆盖已经存在的同名环境变量，便于本地调试。
load_dotenv(override=True)


# init_chat_model 默认参数：不开启采样、最多重试 3 次、单次超时 120 秒。
_DEFAULT_MODEL_KWARGS = {
    "temperature": 0,
    "max_retries": 3,
    "timeout": 120,
}


def _build_model(spec: str, *, provider: str | None = None, **kwargs):
    """构造一个 chat 模型实例。``spec`` 可以是 ``"provider:name"``，也可以只是 ``"name"``。"""
    # 若没有显式指定 provider，但 spec 里带了 "provider:name" 的前缀，自动拆分
    if provider is None and ":" in spec:
        provider, spec = spec.split(":", 1)
    if provider is None:
        provider = "openai"
    # 用户传入的 kwargs 优先级更高，可以覆盖默认参数（例如调高 temperature）
    params = {**_DEFAULT_MODEL_KWARGS, **kwargs}
    return init_chat_model(spec, model_provider=provider, **params)


class _ModelProxy:
    """转发所有属性访问到内部可替换的 chat 模型。

    模块在启动时只 import 一次 ``model``；之后通过 ``set_model`` 替换 proxy 内部
    backend，从而让所有已存在的引用都指向新模型，避免“切换后还在用旧模型”的坑。
    """

    # 用 __slots__ 限制属性集，避免被 __setattr__ 拦截误写到外层
    __slots__ = ("_inner", "_spec", "_provider")

    def __init__(self, inner, spec: str, provider: str):
        # 直接绕过 __setattr__，把内部三个字段写到对象上
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_provider", provider)

    def _swap(self, inner, spec: str, provider: str) -> None:
        # 热替换：原地修改 _inner 引用，外部所有持有 proxy 的人都会立即看到新 backend
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_provider", provider)

    def __getattr__(self, name):
        # 任何未知属性的访问都转发到真实 backend（如 invoke / bind_tools / stream 等）
        return getattr(self._inner, name)

    def invoke(self, *args, **kwargs):
        args, kwargs = _inject_audit_callback(args, kwargs)
        return self._inner.invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        args, kwargs = _inject_audit_callback(args, kwargs)
        return self._inner.stream(*args, **kwargs)

    def bind_tools(self, *args, **kwargs):
        return _AuditedRunnable(self._inner.bind_tools(*args, **kwargs))

    def with_structured_output(self, *args, **kwargs):
        return _AuditedRunnable(self._inner.with_structured_output(*args, **kwargs))

    def __setattr__(self, name, value):
        # 写属性也转发给 backend，保持代理透明
        setattr(self._inner, name, value)

    def __repr__(self):
        # 调试时清楚地看到当前实际使用的 provider:spec
        return f"<ModelProxy {self._provider}:{self._spec}>"


class _AuditedRunnable:
    """Wrap derived runnables so their invoke/stream calls keep audit callbacks."""

    __slots__ = ("_inner",)

    def __init__(self, inner):
        self._inner = inner

    def invoke(self, *args, **kwargs):
        args, kwargs = _inject_audit_callback(args, kwargs)
        return self._inner.invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        args, kwargs = _inject_audit_callback(args, kwargs)
        return self._inner.stream(*args, **kwargs)

    def bind_tools(self, *args, **kwargs):
        return _AuditedRunnable(self._inner.bind_tools(*args, **kwargs))

    def with_structured_output(self, *args, **kwargs):
        return _AuditedRunnable(self._inner.with_structured_output(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _inject_audit_callback(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """Append the audit callback to a RunnableConfig if one is present."""
    config = None
    if len(args) >= 2:
        config = args[1]
        args = (args[0], with_llm_audit_callback(config, model_label=current_model_label()), *args[2:])
        return args, kwargs
    config = kwargs.get("config")
    kwargs = dict(kwargs)
    kwargs["config"] = with_llm_audit_callback(config, model_label=current_model_label())
    return args, kwargs


# 启动时的默认模型，可以通过 /model 命令运行时替换
_INITIAL_SPEC = "gpt-5.4"
_INITIAL_PROVIDER = "openai"

# 模块级单例：所有节点 / 工具都直接 `from .load import model` 使用它。
model = _ModelProxy(
    _build_model(_INITIAL_SPEC, provider=_INITIAL_PROVIDER),
    _INITIAL_SPEC,
    _INITIAL_PROVIDER,
)


def set_model(spec: str, **kwargs) -> str:
    """原地替换底层 chat 模型，返回新的 ``provider:name`` 标签。"""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("模型名不能为空")
    # 优先尊重显式传入的 provider，其次解析 "provider:name" 格式
    provider = kwargs.pop("provider", None)
    if provider is None and ":" in spec:
        provider, spec = spec.split(":", 1)
    if provider is None:
        provider = "openai"
    new_inner = _build_model(spec, provider=provider, **kwargs)
    # 替换 proxy 内部 backend；所有已 import 的 model 引用自动指向新模型
    model._swap(new_inner, spec, provider)
    return f"{provider}:{spec}"


def current_model_label() -> str:
    """返回当前活跃模型的可读标签（用于状态展示等场景）。"""
    return f"{model._provider}:{model._spec}"


# Tavily 联网搜索客户端：tavily_search 工具会调用它
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# OpenWeather API Key：check_weather 工具按需读取，未配置时给出友好提示
openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
