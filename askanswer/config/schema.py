"""``Settings``：AskAnswer 的全部运行配置，一处定义、有类型、可注入。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

DigestMode = Literal["off", "brief", "llm"]
ROLE_NAMES = ("answer", "classify", "evaluate", "summarize")


class ModelRoles(BaseModel):
    """按角色路由的模型标签；None 表示跟随全局默认模型。"""

    model_config = ConfigDict(frozen=True)

    answer: str | None = None
    classify: str | None = None
    evaluate: str | None = None
    summarize: str | None = None
    fallbacks: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    def spec(self, role: str) -> str | None:
        return getattr(self, role, None)

    def fallback_chain(self, role: str) -> tuple[str, ...]:
        return tuple(self.fallbacks.get(role, ()))


class Limits(BaseModel):
    model_config = ConfigDict(frozen=True)

    read_file_max_bytes: int = Field(10 * 1024 * 1024, gt=0)
    read_file_max_chars: int = Field(60_000, gt=0)
    write_file_max_bytes: int = Field(1024 * 1024, gt=0)
    shell_output_max_bytes: int = Field(64 * 1024, gt=0)
    http_timeout_seconds: float = Field(10, gt=0)
    search_timeout_seconds: float = Field(30, gt=0)


class Secrets(BaseModel):
    """第三方 API key；用 SecretStr，repr / 日志里不会泄漏明文。"""

    model_config = ConfigDict(frozen=True)

    tavily_api_key: SecretStr | None = None
    openweather_api_key: SecretStr | None = None
    server_token: SecretStr | None = None

    @staticmethod
    def reveal(value: SecretStr | None) -> str | None:
        return value.get_secret_value() if value is not None else None


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_model: str = "openai:gpt-5.4"
    models: ModelRoles = Field(default_factory=ModelRoles)
    tenant_id: str | None = None
    db_path: str | None = None          # None → 数据目录下的 state.db
    db_dsn: str | None = None           # SQL 子 Agent 默认连接串
    db_dialect: str | None = None
    mcp_profile: str | None = None      # None → 数据目录下的 mcp.json
    mcp_all_intents: bool = False
    deny_tools: frozenset[str] = frozenset()
    context_max_tokens: int | None = Field(None, gt=0)
    context_digest: DigestMode = "off"
    run_token_budget: int | None = Field(None, gt=0)
    data_home: str | None = None        # XDG_DATA_HOME
    otel_exporter: str | None = None    # ASKANSWER_OTEL_EXPORTER
    limits: Limits = Field(default_factory=Limits)
    secrets: Secrets = Field(default_factory=Secrets)
