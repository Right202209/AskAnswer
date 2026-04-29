from __future__ import annotations

import os
from functools import lru_cache

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool, StructuredTool

from ..load import model
from ..schema import ContextSchema, normalize_context


_DSN_ENV_VAR = "WLANGGRAPH_POSTGRES_DSN"


def get_database_uri(context: ContextSchema | dict | None = None) -> str:
    runtime_context = normalize_context(context)
    database_uri = (runtime_context.db_dsn or os.getenv(_DSN_ENV_VAR, "")).strip()
    if not database_uri:
        raise RuntimeError(f"未配置 runtime.context.db_dsn 或 {_DSN_ENV_VAR}")
    return database_uri


@lru_cache(maxsize=16)
def _get_database(database_uri: str) -> SQLDatabase:
    return SQLDatabase.from_uri(database_uri)


def get_database(context: ContextSchema | dict | None = None) -> SQLDatabase:
    return _get_database(get_database_uri(context))


@lru_cache(maxsize=16)
def _get_toolkit(database_uri: str) -> SQLDatabaseToolkit:
    return SQLDatabaseToolkit(db=_get_database(database_uri), llm=model)


def get_toolkit(context: ContextSchema | dict | None = None) -> SQLDatabaseToolkit:
    return _get_toolkit(get_database_uri(context))


@lru_cache(maxsize=16)
def _get_sql_tools(database_uri: str) -> tuple[BaseTool, ...]:
    database = _get_database(database_uri)
    toolkit = _get_toolkit(database_uri)
    return (
        *toolkit.get_tools(),
        _make_get_schema_tool(database),
        _make_find_slow_sql_tool(database),
    )


def get_sql_tools(context: ContextSchema | dict | None = None) -> tuple[BaseTool, ...]:
    return _get_sql_tools(get_database_uri(context))


def get_sql_tool(name: str, context: ContextSchema | dict | None = None) -> BaseTool:
    for sql_tool in get_sql_tools(context):
        if sql_tool.name == name:
            return sql_tool
    raise RuntimeError(f"未找到 SQL 工具：{name}")


def get_sql_dialect(context: ContextSchema | dict | None = None) -> str:
    runtime_context = normalize_context(context)
    if runtime_context.db_dialect:
        return runtime_context.db_dialect
    return str(get_database(runtime_context).dialect)


def _make_get_schema_tool(database: SQLDatabase) -> BaseTool:
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
    def find_slow_sql(_: str = "") -> str:
        """查看数据库中的慢 SQL 统计（PostgreSQL pg_stat_statements）。"""

        if "postgres" not in str(database.dialect).lower():
            return "find_slow_sql 仅支持 PostgreSQL。"

        query = """
        SELECT query, calls, total_exec_time, mean_exec_time
        FROM pg_stat_statements
        ORDER BY mean_exec_time DESC
        LIMIT 20;
        """.strip()

        try:
            result = database.run(query, fetch="all")
        except Exception as exc:
            return f"查询慢 SQL 失败：{exc}"

        return str(result)

    return StructuredTool.from_function(
        func=find_slow_sql,
        name="find_slow_sql",
        description="查看数据库中的慢 SQL 统计（PostgreSQL pg_stat_statements）。",
    )
