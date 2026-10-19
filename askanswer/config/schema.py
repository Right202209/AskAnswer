"""Per-invocation runtime context (LangGraph ``context_schema``).

Unlike ``SearchState``, :class:`ContextSchema` is not checkpointed or merged: it
is passed read-only to every node and tool (``Runtime[ContextSchema]`` /
``ToolRuntime[ContextSchema]``) for each call.
(与 SearchState 不同，ContextSchema 按调用作为只读配置传给节点 / 工具。)
"""
from dataclasses import dataclass, fields


@dataclass
class ContextSchema:
    """Runtime configuration for one graph invocation.

    :ivar llm_provider: default LLM provider label (``"openai"``).
    :ivar db_dsn: database URL for the SQL subagent (SQLAlchemy URL).
    :ivar db_dialect: optional dialect hint for SQL generation (e.g. ``"postgresql"``).
    :ivar tenant_id: tenant used for audit / thread metadata and SQL connection isolation.
    """

    # 选用的 LLM 提供方，目前主要为 "openai"
    llm_provider: str = "openai"
    # 数据库连接串：SQL agent 通过宿主传入的运行时上下文读取
    db_dsn: str | None = None
    # 数据库方言：例如 "postgresql"、"mysql"，用于 prompt 中提示生成对应方言的 SQL
    db_dialect: str | None = None
    # 多租户隔离用的租户 ID（可选）
    tenant_id: str | None = None


def normalize_context(context: ContextSchema | dict | None) -> ContextSchema:
    """Coerce ``context`` (``ContextSchema`` / ``dict`` / ``None``) into a ``ContextSchema``.

    Unknown dict keys are ignored. (把外部传入的 context 统一转成 ContextSchema 实例。)

    LangGraph 在不同版本下有时会把 context 当成 dict 传进来，所以这里做一层兼容：
    - 已经是 ContextSchema：直接返回；
    - 是 dict：仅保留 ContextSchema 已声明的字段，丢弃多余 key 后构造；
    - 其它（None 等）：返回字段全为默认值的 ContextSchema。
    """
    if isinstance(context, ContextSchema):
        return context
    if isinstance(context, dict):
        # 过滤掉不在 ContextSchema 中声明的键，避免 TypeError
        valid_fields = {field.name for field in fields(ContextSchema)}
        values = {key: value for key, value in context.items() if key in valid_fields}
        return ContextSchema(**values)
    return ContextSchema()
