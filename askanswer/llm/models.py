# 全局模型/外部客户端加载与热替换。
# - 通过 _ModelProxy 让所有模块只在启动时导入一次 model，
#   /model 切换时只替换 proxy 内部的 backend，所有引用自动指向新模型。
# - 同时初始化 Tavily 搜索客户端、读取 OpenWeather Key 等运行时常量。
from langchain.chat_models import init_chat_model

from askanswer.config import ensure_environ
from askanswer.observability.audit import with_llm_audit_callback

# 配置装载见 ``askanswer.config.ensure_environ``（入口显式调用，P1/P3）。


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


def build_backend(spec: str, **kwargs):
    """按 "provider:name" 标签构造 backend；供 routing 层解析回退候选复用。

    与 ``_build_model`` 的差别仅是入参形态：这里统一接收带 provider 前缀的标签，
    routing 层不必关心 provider 拆分细节。
    """
    return _build_model(spec, **kwargs)


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

    def _ensure_spec(self) -> None:
        # 默认模型标签也惰性解析：要等 ensure_environ() 之后 ASKANSWER_DEFAULT_MODEL 才可信
        if self._spec is None:
            ensure_environ()
            spec, provider = _initial_model()
            object.__setattr__(self, "_spec", spec)
            object.__setattr__(self, "_provider", provider)

    def _ensure_inner(self):
        # 惰性构造：首次真正调用模型时才 init_chat_model。
        # 约束：`import askanswer.*` 不得要求 OPENAI_API_KEY（spec §5 冒烟矩阵第 1 项）。
        if self._inner is None:
            self._ensure_spec()
            object.__setattr__(
                self, "_inner", _build_model(self._spec, provider=self._provider)
            )
        return self._inner

    def __getattr__(self, name):
        # 任何未知属性的访问都转发到真实 backend（如 invoke / bind_tools / stream 等）
        return getattr(self._ensure_inner(), name)

    def invoke(self, *args, **kwargs):
        args, kwargs = inject_llm_callbacks(args, kwargs)
        return self._ensure_inner().invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        args, kwargs = inject_llm_callbacks(args, kwargs)
        return self._ensure_inner().stream(*args, **kwargs)

    def bind_tools(self, *args, **kwargs):
        return _AuditedRunnable(self._ensure_inner().bind_tools(*args, **kwargs))

    def with_structured_output(self, *args, **kwargs):
        return _AuditedRunnable(self._ensure_inner().with_structured_output(*args, **kwargs))

    def __setattr__(self, name, value):
        # 写属性也转发给 backend，保持代理透明
        setattr(self._ensure_inner(), name, value)

    def __repr__(self):
        # 调试时清楚地看到当前实际使用的 provider:spec
        return f"<ModelProxy {self.label}>"

    @property
    def label(self) -> str:
        self._ensure_spec()
        return f"{self._provider}:{self._spec}"


class _AuditedRunnable:
    """Wrap derived runnables so their invoke/stream calls keep audit callbacks."""

    __slots__ = ("_inner",)

    def __init__(self, inner):
        self._inner = inner

    def invoke(self, *args, **kwargs):
        args, kwargs = inject_llm_callbacks(args, kwargs)
        return self._inner.invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        args, kwargs = inject_llm_callbacks(args, kwargs)
        return self._inner.stream(*args, **kwargs)

    def bind_tools(self, *args, **kwargs):
        return _AuditedRunnable(self._inner.bind_tools(*args, **kwargs))

    def with_structured_output(self, *args, **kwargs):
        return _AuditedRunnable(self._inner.with_structured_output(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._inner, name)


def inject_llm_callbacks(
    args: tuple,
    kwargs: dict,
    *,
    label: str | None = None,
) -> tuple[tuple, dict]:
    """Append the audit (and, if enabled, telemetry) callback to a RunnableConfig.

    ``label=None`` 归因到当前全局模型；``routing.RoutedModel`` 传显式 label，
    确保回退到另一个模型后 token 用量仍归因到真正执行调用的那一个。
    """
    resolved = label or current_model_label()
    if len(args) >= 2:
        config = _augment_config(args[1], resolved)
        return (args[0], config, *args[2:]), kwargs
    kwargs = dict(kwargs)
    kwargs["config"] = _augment_config(kwargs.get("config"), resolved)
    return args, kwargs


def _augment_config(config, model_label: str) -> dict:
    """Compose audit + telemetry callbacks onto one config (independent write paths)."""
    cfg = with_llm_audit_callback(config, model_label=model_label)
    telemetry_cb = _telemetry_callback(model_label)
    if telemetry_cb is not None:
        # with_llm_audit_callback 保证 cfg["callbacks"] 是一个全新的可变 list。
        cfg["callbacks"].append(telemetry_cb)
    return cfg


def _telemetry_callback(model_label: str):
    """Lazy import telemetry：顶层不依赖它，且未启用时返回 None（零开销）。"""
    from askanswer.observability import telemetry

    return telemetry.llm_callback(model_label)


# 启动时的默认模型，可以通过 /model 命令运行时替换。
# settings.json 的 "model" / env ASKANSWER_DEFAULT_MODEL 可覆盖默认值。
def _initial_model() -> tuple[str, str]:
    from askanswer.config import current_settings

    raw = current_settings().default_model.strip()
    if ":" in raw:
        provider, spec = raw.split(":", 1)
        provider, spec = provider.strip() or "openai", spec.strip()
        if spec:
            return spec, provider
    return (raw or "gpt-5.4"), "openai"


# 模块级单例：所有节点 / 工具都直接 `from .load import model` 使用它。
# inner/spec 均为 None → 惰性构造：首次调用（invoke/stream/bind_tools…）时才解析
# 默认模型并建模型，使得纯导入（`--graph`、测试、helix_mcp）不读配置、无需 API key。
model = _ModelProxy(None, None, None)


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
    return model.label


def raw_backend():
    """返回全局代理**当前**的真实 backend（不带审计包裹）。

    routing 层用它作为「跟随全局 `/model`」角色的候选：拿到的是热替换后的
    最新 backend，回退编排会自己补上审计 callback，避免双重注入。

    经 ``_ensure_inner``：惰性初始化未触发时也会构造 backend，避免返回 None。
    """
    return model._ensure_inner()


_tavily_client = None


def get_tavily_client():
    """惰性构造 Tavily 客户端：导入本模块不建任何外部客户端（架构文档 P1）。"""
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient

        ensure_environ()
        from askanswer.config import current_settings
        from askanswer.config.schema import Secrets

        api_key = Secrets.reveal(current_settings().secrets.tavily_api_key)
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def openweather_api_key() -> str | None:
    """按需读取 OpenWeather Key，未配置时 check_weather 给出友好提示。"""
    from askanswer.config import current_settings
    from askanswer.config.schema import Secrets

    return Secrets.reveal(current_settings().secrets.openweather_api_key)
