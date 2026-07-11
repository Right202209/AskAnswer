# SQL 子图所需的数据库交互层：连接、toolkit、扩展工具的缓存与封装。
# 关键点：所有缓存都按 DSN 走 lru_cache，避免每次调用都重新连数据库。
from __future__ import annotations

import os
from functools import lru_cache

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool, StructuredTool

from ..load import model
from ..schema import ContextSchema, normalize_context


# 兼容旧环境变量名：当 ContextSchema.db_dsn 没传时回退到该环境变量
_DSN_ENV_VAR = "WLANGGRAPH_POSTGRES_DSN"


def get_database_uri(context: ContextSchema | dict | None = None) -> str:
    """优先取 context.db_dsn；没有则从环境变量取；都没有就抛错。"""
    runtime_context = normalize_context(context)
    database_uri = (runtime_context.db_dsn or os.getenv(_DSN_ENV_VAR, "")).strip()
    if not database_uri:
        raise RuntimeError(f"未配置 runtime.context.db_dsn 或 {_DSN_ENV_VAR}")
    return database_uri


def _cache_key(context: ContextSchema | dict | None = None) -> tuple[str | None, str]:
    """连接缓存 key：``(tenant_id, dsn)``。

    刻意把 tenant_id 纳入 key —— 即便两个租户配了相同 DSN，也各自持有独立的
    SQLDatabase / toolkit / 工具实例，不共享连接池，避免跨租户串数据或串会话级
    连接状态（对齐 execution-plan Phase 1.3：不同 tenant 不复用 connection cache）。
    """
    runtime_context = normalize_context(context)
    return runtime_context.tenant_id, get_database_uri(runtime_context)


@lru_cache(maxsize=16)
def _get_database(cache_key: tuple[str | None, str]) -> SQLDatabase:
    """按 (tenant, DSN) 缓存 SQLDatabase 实例（连接池在内部维护）。"""
    return SQLDatabase.from_uri(cache_key[1])


def get_database(context: ContextSchema | dict | None = None) -> SQLDatabase:
    return _get_database(_cache_key(context))


@lru_cache(maxsize=16)
def _get_toolkit(cache_key: tuple[str | None, str]) -> SQLDatabaseToolkit:
    """按 (tenant, DSN) 缓存 LangChain 自带的 SQLDatabaseToolkit。"""
    return SQLDatabaseToolkit(db=_get_database(cache_key), llm=model)


def get_toolkit(context: ContextSchema | dict | None = None) -> SQLDatabaseToolkit:
    return _get_toolkit(_cache_key(context))


@lru_cache(maxsize=16)
def _get_sql_tools(cache_key: tuple[str | None, str]) -> tuple[BaseTool, ...]:
    """组合 LangChain 自带工具 + 我们扩展的两个工具，整体缓存。"""
    database = _get_database(cache_key)
    toolkit = _get_toolkit(cache_key)
    return (
        *toolkit.get_tools(),
        # 自带 toolkit 的 schema 工具入参较繁琐，封装一个简单版
        _make_get_schema_tool(database),
        # PostgreSQL 专属：基于 pg_stat_statements 的慢 SQL 排行
        _make_find_slow_sql_tool(database),
    )


def get_sql_tools(context: ContextSchema | dict | None = None) -> tuple[BaseTool, ...]:
    return _get_sql_tools(_cache_key(context))


def get_sql_tool(name: str, context: ContextSchema | dict | None = None) -> BaseTool:
    """按工具名取出某个具体的 SQL 工具，找不到时抛错。"""
    for sql_tool in get_sql_tools(context):
        if sql_tool.name == name:
            return sql_tool
    raise RuntimeError(f"未找到 SQL 工具：{name}")


def get_sql_dialect(context: ContextSchema | dict | None = None) -> str:
    """返回当前数据库方言名（如 postgresql/mysql/sqlite），优先用 context 里显式声明的。"""
    runtime_context = normalize_context(context)
    if runtime_context.db_dialect:
        return runtime_context.db_dialect
    # context 没声明就从 SQLDatabase 实例反查
    return str(get_database(runtime_context).dialect)


def _make_get_schema_tool(database: SQLDatabase) -> BaseTool:
    """构造一个简化版 get_schema 工具：调用 database.get_table_info 即可。"""
    def get_schema(_: str = "") -> str:
        """获取数据库表结构，供 SQL 生成使用。"""

        table_names = list(database.get_usable_table_names())
        if not table_names:
            return "数据库中没有可用表。"
        return database.get_table_info(table_names)

    return StructuredTool.from_function(
        func=get_schema,
        name="get_schema",
        description="获取数据库表结构，供 SQL 生成使用。",
    )


def _make_find_slow_sql_tool(database: SQLDatabase) -> BaseTool:
    """构造慢 SQL 排行工具，仅在 PostgreSQL 上有意义。"""
    def find_slow_sql(_: str = "") -> str:
        """查看数据库中的慢 SQL 统计（PostgreSQL pg_stat_statements）。"""

        # 非 PostgreSQL 直接返回友好提示，避免向其它方言下发不兼容的 SQL
        if "postgres" not in str(database.dialect).lower():
            return "find_slow_sql 仅支持 PostgreSQL。"

        # 按平均执行时间排序的慢 SQL Top 20
        query = """
        SELECT query, calls, total_exec_time, mean_exec_time
        FROM pg_stat_statements
        ORDER BY mean_exec_time DESC
        LIMIT 20;
        """.strip()

        try:
            result = database.run(query, fetch="all")
        except Exception as exc:
            # 通常是没装 pg_stat_statements 扩展或权限不足
            return f"查询慢 SQL 失败：{exc}"

        return str(result)

    return StructuredTool.from_function(
        func=find_slow_sql,
        name="find_slow_sql",
        description="查看数据库中的慢 SQL 统计（PostgreSQL pg_stat_statements）。",
    )
