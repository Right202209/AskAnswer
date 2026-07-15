"""按「角色」路由模型：不同任务用不同模型，并支持跨 provider 故障回退。

为什么按角色而不是按 intent 路由：intent 决定「做什么」，角色决定「这一步
LLM 调用的形状」。同一个 intent 里会出现多种调用形状 —— 分类是短输入 +
结构化输出，评估是中输入 + 一行裁决，摘要是长输入 + 短输出，只有主回答
需要旗舰模型的推理与工具调用能力。按角色路由让「小模型跑便宜活」成为
默认可配置项，而不是散落在各节点里的硬编码。

设计约定（对齐 docs/important-documentation-d1-routing-context-cost-eval.md）：
- 默认零回归：不设任何环境变量时，所有角色都解析为全局 ``_ModelProxy``，
  行为与引入本模块之前完全一致；`/model` 热替换继续对全体生效。
- 回退链只做「换模型重试」，不做退避重试 —— 单模型内的网络重试已由
  ``init_chat_model(max_retries=3)`` 承担，这里解决的是 provider 级故障。
- 每次回退写一条 audit 事件（kind="model_fallback"），保证可观测、可复盘。
- token 用量归因到真正执行调用的那个模型标签（见 ``inject_llm_callbacks``）。

环境变量（值均为 "provider:name"，缺 provider 前缀时默认 openai）：
- ``ASKANSWER_MODEL_ANSWER`` / ``_CLASSIFY`` / ``_EVALUATE`` / ``_SUMMARIZE``
- ``ASKANSWER_MODEL_FALLBACKS_<ROLE>``：逗号分隔的回退模型列表
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .audit import log_event
from .load import (
    build_backend,
    current_model_label,
    inject_llm_callbacks,
    model,
    raw_backend,
)

ROLE_ANSWER = "answer"        # react 主推理：工具调用 + 最终回答（旗舰）
ROLE_CLASSIFY = "classify"    # 意图分类：短输入 + 结构化输出（小模型足够）
ROLE_EVALUATE = "evaluate"    # sorcery 质量评估：LLM-as-judge（中/小模型）
ROLE_SUMMARIZE = "summarize"  # 历史滚动摘要：长输入 + 短输出（小模型）
ROLES = (ROLE_ANSWER, ROLE_CLASSIFY, ROLE_EVALUATE, ROLE_SUMMARIZE)

_ROLE_ENV_PREFIX = "ASKANSWER_MODEL_"
_FALLBACK_ENV_PREFIX = "ASKANSWER_MODEL_FALLBACKS_"
_DEFAULT_PROVIDER = "openai"
# 已构建 backend 的缓存上限：角色 × 回退链的组合有限，8 个足够覆盖
_BACKEND_CACHE_MAX = 8


def normalize_spec(spec: str) -> str:
    """把 "name" / "provider:name" 统一成 "provider:name" 标签形式。"""
    clean = str(spec or "").strip()
    if ":" in clean:
        return clean
    return f"{_DEFAULT_PROVIDER}:{clean}"


@dataclass(frozen=True)
class ModelRoute:
    """一个角色的路由：主模型 + 回退链。``spec=None`` 表示跟随全局 `/model`。"""

    role: str
    spec: str | None
    fallbacks: tuple[str, ...] = ()

    @property
    def is_default(self) -> bool:
        return self.spec is None and not self.fallbacks


class ModelRouter:
    """角色 → 路由表。由 ``get_router()`` 在首次使用时从环境变量构建。"""

    def __init__(self, routes: dict[str, ModelRoute]):
        self._routes = routes

    def route(self, role: str) -> ModelRoute:
        return self._routes.get(role) or ModelRoute(role=role, spec=None)

    def describe(self) -> dict[str, str]:
        """人类可读的路由表快照（诊断 / 评测报告用）。"""
        out: dict[str, str] = {}
        for role in ROLES:
            route = self.route(role)
            primary = route.spec or f"(全局 {current_model_label()})"
            chain = " → ".join((primary, *route.fallbacks))
            out[role] = chain
        return out


def _route_from_env(role: str) -> ModelRoute:
    spec = os.environ.get(f"{_ROLE_ENV_PREFIX}{role.upper()}", "").strip()
    raw_fallbacks = os.environ.get(f"{_FALLBACK_ENV_PREFIX}{role.upper()}", "")
    fallbacks = tuple(
        normalize_spec(item) for item in raw_fallbacks.split(",") if item.strip()
    )
    return ModelRoute(
        role=role,
        spec=normalize_spec(spec) if spec else None,
        fallbacks=fallbacks,
    )


_router: ModelRouter | None = None


def get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter({role: _route_from_env(role) for role in ROLES})
    return _router


def reset_router() -> None:
    """清掉路由表缓存（环境变量变化 / 单测隔离时用，下次访问重新读 env）。"""
    global _router
    _router = None


# ── backend 解析与缓存 ────────────────────────────────────────────────

_BACKEND_CACHE: dict[str, object] = {}


def _cached_backend(spec: str) -> tuple[str, object]:
    label = normalize_spec(spec)
    hit = _BACKEND_CACHE.get(label)
    if hit is not None:
        return label, hit
    backend = build_backend(label)
    if len(_BACKEND_CACHE) >= _BACKEND_CACHE_MAX:
        # 简单 FIFO 淘汰：路由组合本来就少，不值得引入 LRU 依赖
        _BACKEND_CACHE.pop(next(iter(_BACKEND_CACHE)))
    _BACKEND_CACHE[label] = backend
    return label, backend


def _resolve_backend(spec: str | None) -> tuple[str, object]:
    """spec=None → 全局代理**当前**的 backend（`/model` 热替换后自动跟随）。"""
    if spec is None:
        return current_model_label(), raw_backend()
    return _cached_backend(spec)


def _apply_transforms(backend: object, transforms: tuple) -> object:
    runnable = backend
    for name, args, kwargs in transforms:
        runnable = getattr(runnable, name)(*args, **kwargs)
    return runnable


def _log_fallback(*, role: str, label: str, error: Exception) -> None:
    log_event(
        kind="model_fallback",
        model_label=label,
        args_summary=f"role={role}",
        error=str(error),
    )


_STREAM_EMPTY = object()


def _resume_stream(first, rest):
    """把「已取出的首个分片」和剩余迭代器重新拼成一个流。"""
    if first is _STREAM_EMPTY:
        return
    yield first
    yield from rest


class RoutedModel:
    """带回退链的模型句柄，接口与 ``_ModelProxy`` 对齐。

    ``bind_tools`` / ``with_structured_output`` 不立即构造 runnable，而是把
    变换记录下来、调用时对每个候选 backend 重放 —— 回退到另一家 provider
    时工具绑定 / 结构化输出同样生效。
    """

    __slots__ = ("_route", "_transforms")

    def __init__(self, route: ModelRoute, transforms: tuple = ()):
        self._route = route
        self._transforms = transforms

    @property
    def label(self) -> str:
        if self._route.spec is None:
            return current_model_label()
        return normalize_spec(self._route.spec)

    def bind_tools(self, *args, **kwargs) -> "RoutedModel":
        record = ("bind_tools", args, kwargs)
        return RoutedModel(self._route, (*self._transforms, record))

    def with_structured_output(self, *args, **kwargs) -> "RoutedModel":
        record = ("with_structured_output", args, kwargs)
        return RoutedModel(self._route, (*self._transforms, record))

    def _candidates(self):
        """依次产出 (label, runnable)；构建失败的候选记 fallback 事件后跳过。

        构建失败最常见的原因是回退目标的 SDK 未安装（如 langchain-anthropic）
        或 spec 拼写错误 —— 这类问题应该降级到下一候选，而不是让回答崩掉。
        """
        for spec in (self._route.spec, *self._route.fallbacks):
            try:
                label, backend = _resolve_backend(spec)
                runnable = _apply_transforms(backend, self._transforms)
            except Exception as exc:
                _log_fallback(role=self._route.role, label=str(spec), error=exc)
                continue
            yield label, runnable

    def invoke(self, *args, **kwargs):
        last_error: Exception | None = None
        for label, runnable in self._candidates():
            call_args, call_kwargs = inject_llm_callbacks(args, kwargs, label=label)
            try:
                return runnable.invoke(*call_args, **call_kwargs)
            except Exception as exc:
                last_error = exc
                _log_fallback(role=self._route.role, label=label, error=exc)
        raise last_error or RuntimeError(f"角色 {self._route.role} 没有可用的模型候选")

    def stream(self, *args, **kwargs):
        """流式调用：仅在**首个分片产出前**失败才回退，避免向用户重复输出。"""
        last_error: Exception | None = None
        for label, runnable in self._candidates():
            call_args, call_kwargs = inject_llm_callbacks(args, kwargs, label=label)
            try:
                iterator = runnable.stream(*call_args, **call_kwargs)
                first = next(iterator, _STREAM_EMPTY)
            except Exception as exc:
                last_error = exc
                _log_fallback(role=self._route.role, label=label, error=exc)
                continue
            return _resume_stream(first, iterator)
        raise last_error or RuntimeError(f"角色 {self._route.role} 没有可用的模型候选")


def model_for(role: str):
    """取该角色应使用的模型句柄。

    未配置任何路由时直接返回全局 ``_ModelProxy``（零行为变化）；配置了主模型
    或回退链时返回 ``RoutedModel``。两者都支持 invoke / stream / bind_tools /
    with_structured_output，调用方无需区分。
    """
    route = get_router().route(role)
    if route.is_default:
        return model
    return RoutedModel(route)


def describe_routes() -> dict[str, str]:
    """当前生效的角色路由表（`/status`、评测报告等诊断场景用）。"""
    return get_router().describe()
