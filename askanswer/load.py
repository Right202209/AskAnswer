import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAI
from tavily import TavilyClient

load_dotenv(override=True)


_DEFAULT_MODEL_KWARGS = {
    "temperature": 0,
    "max_retries": 3,
    "timeout": 120,
}


def _build_model(spec: str, *, provider: str | None = None, **kwargs):
    """Create a chat model. ``spec`` may be ``"provider:name"`` or just ``"name"``."""
    if provider is None and ":" in spec:
        provider, spec = spec.split(":", 1)
    if provider is None:
        provider = "openai"
    params = {**_DEFAULT_MODEL_KWARGS, **kwargs}
    return init_chat_model(spec, model_provider=provider, **params)


class _ModelProxy:
    """Forward all attribute access to a swappable underlying chat model.

    Modules import ``model`` once at startup; this proxy lets ``set_model``
    redirect every existing reference to a freshly built backend.
    """

    __slots__ = ("_inner", "_spec", "_provider")

    def __init__(self, inner, spec: str, provider: str):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_provider", provider)

    def _swap(self, inner, spec: str, provider: str) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_provider", provider)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)

    def __repr__(self):
        return f"<ModelProxy {self._provider}:{self._spec}>"


_INITIAL_SPEC = "gpt-5.4"
_INITIAL_PROVIDER = "openai"

model = _ModelProxy(
    _build_model(_INITIAL_SPEC, provider=_INITIAL_PROVIDER),
    _INITIAL_SPEC,
    _INITIAL_PROVIDER,
)


def set_model(spec: str, **kwargs) -> str:
    """Swap the underlying chat model in place. Returns the new ``provider:name``."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("模型名不能为空")
    provider = kwargs.pop("provider", None)
    if provider is None and ":" in spec:
        provider, spec = spec.split(":", 1)
    if provider is None:
        provider = "openai"
    new_inner = _build_model(spec, provider=provider, **kwargs)
    model._swap(new_inner, spec, provider)
    return f"{provider}:{spec}"


def current_model_label() -> str:
    """Human-readable label for the currently active model."""
    return f"{model._provider}:{model._spec}"


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
