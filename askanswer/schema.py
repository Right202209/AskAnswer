from dataclasses import dataclass, fields
from langgraph.graph import StateGraph

from .state import SearchState


@dataclass
class ContextSchema:
    llm_provider: str = "openai"
    db_dsn: str | None = None
    db_dialect: str = "postgres"
    tenant_id: str | None = None


def normalize_context(context: ContextSchema | dict | None) -> ContextSchema:
    if isinstance(context, ContextSchema):
        return context
    if isinstance(context, dict):
        valid_fields = {field.name for field in fields(ContextSchema)}
        values = {key: value for key, value in context.items() if key in valid_fields}
        return ContextSchema(**values)
    return ContextSchema()

graph = StateGraph(SearchState, context_schema=ContextSchema)
