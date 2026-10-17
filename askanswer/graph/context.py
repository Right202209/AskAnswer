# LangGraph 的“运行时上下文”定义。
# 与 SearchState 不同，ContextSchema 不会进入消息合并流程，
# 而是按调用作为只读配置传给节点 / 工具（通过 Runtime[ContextSchema]）。
from dataclasses import dataclass, fields


@dataclass
class ContextSchema:
    # 选用的 LLM 提供方，目前主要为 "openai"
    llm_provider: str = "openai"
    # 数据库连接串：SQL agent 需要使用，CLI 会从环境变量读取后注入
    db_dsn: str | None = None
    # 数据库方言：例如 "postgresql"、"mysql"，用于 prompt 中提示生成对应方言的 SQL
    db_dialect: str | None = None
    # 多租户隔离用的租户 ID（可选）
    tenant_id: str | None = None


def normalize_context(context: ContextSchema | dict | None) -> ContextSchema:
    """把外部传入的 context 统一转成 ContextSchema 实例。

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
