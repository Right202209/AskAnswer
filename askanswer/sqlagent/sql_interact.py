from __future__ import annotations

import os
from functools import lru_cache

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import BaseTool, tool

from ..load import model


_DSN_ENV_VAR = "WLANGGRAPH_POSTGRES_DSN"


def _get_database_uri() -> str:
    database_uri = os.getenv(_DSN_ENV_VAR, "").strip()
    if not database_uri:
        raise RuntimeError(f"未配置 {_DSN_ENV_VAR}")
    return database_uri


@lru_cache(maxsize=1)
def get_database() -> SQLDatabase:
    return SQLDatabase.from_uri(_get_database_uri())


@lru_cache(maxsize=1)
def get_toolkit() -> SQLDatabaseToolkit:
    return SQLDatabaseToolkit(db=get_database(), llm=model)


@lru_cache(maxsize=1)
def get_sql_tools() -> tuple[BaseTool, ...]:
    return (*get_toolkit().get_tools(), get_schema, find_slow_sql)


def get_sql_tool(name: str) -> BaseTool:
    for sql_tool in get_sql_tools():
        if sql_tool.name == name:
            return sql_tool
    raise RuntimeError(f"未找到 SQL 工具：{name}")


def get_sql_dialect() -> str:
    return str(get_database().dialect)


@tool
def get_schema(_: str = "") -> str:
    """获取数据库表结构，供 SQL 生成使用。"""

    database = get_database()
    table_names = list(database.get_usable_table_names())
    if not table_names:
        return "数据库中没有可用表。"
    return database.get_table_info(table_names)


@tool
def find_slow_sql(_: str = "") -> str:
    """查看数据库中的慢 SQL 统计（PostgreSQL pg_stat_statements）。"""

    database = get_database()
    if "postgres" not in get_sql_dialect().lower():
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
