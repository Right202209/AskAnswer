# SQL 子图所需的数据库交互层：连接、SQL 工具集、扩展工具的缓存与封装。
# 关键点：所有缓存都按 (tenant, DSN) 走 lru_cache，避免每次调用都重新连数据库。
# 底层是原生 SQLAlchemy（database.py），不再依赖 langchain-community；
# 工具名 / 入参 / 输出格式与原 SQLDatabaseToolkit 保持一致（子图按名字引用）。
from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ...config.schema import ContextSchema, normalize_context
from ...config.settings import DB_DSN_ENV, resolve_db_dsn
from ...llm.proxy import model
from .database import SqlDatabase


def get_database_uri(context: ContextSchema | dict | None = None) -> str:
    """优先取 context.db_dsn；没有则读 ``ASKANSWER_DB_DSN``（``WLANGGRAPH_POSTGRES_DSN``
    仅作已废弃回退）；都没有就抛错。"""
    runtime_context = normalize_context(context)
    database_uri = (runtime_context.db_dsn or resolve_db_dsn() or "").strip()
    if not database_uri:
        raise RuntimeError(f"未配置 runtime.context.db_dsn 或 {DB_DSN_ENV}")
    return database_uri


def _cache_key(context: ContextSchema | dict | None = None) -> tuple[str | None, str]:
    """连接缓存 key：``(tenant_id, dsn)``。

    刻意把 tenant_id 纳入 key —— 即便两个租户配了相同 DSN，也各自持有独立的
    SQLDatabase / toolkit / 工具实例，不共享连接池，避免跨租户串数据或串会话级
    连接状态（对齐 execution-plan Phase 1.3：不同 tenant 不复用 connection cache）。
    """
    runtime_context = normalize_context(context)
    return runtime_context.tenant_id, get_database_uri(runtime_context)


# 已创建过 engine 的 cache key，供 clear_sql_caches() 释放连接池。
_LIVE_KEYS: set[tuple[str | None, str]] = set()


@lru_cache(maxsize=16)
def _get_database(cache_key: tuple[str | None, str]) -> SqlDatabase:
    """按 (tenant, DSN) 缓存 SqlDatabase 实例（连接池由 SQLAlchemy engine 维护）。"""
    _LIVE_KEYS.add(cache_key)
    return SqlDatabase.from_uri(cache_key[1])


def get_database(context: ContextSchema | dict | None = None) -> SqlDatabase:
    return _get_database(_cache_key(context))


def clear_sql_caches() -> None:
    """释放所有缓存的 engine / 工具（测试隔离、DSN 轮换时用）。"""
    for key in list(_LIVE_KEYS):
        try:
            _get_database(key).dispose()
        except Exception:
            pass
    _LIVE_KEYS.clear()
    _get_database.cache_clear()
    _get_sql_tools.cache_clear()


@lru_cache(maxsize=16)
def _get_sql_tools(cache_key: tuple[str | None, str]) -> tuple[BaseTool, ...]:
    """SQL 工具集（与原 SQLDatabaseToolkit 同名同语义）+ 两个扩展工具，整体缓存。"""
    database = _get_database(cache_key)
    return (
        *_make_toolkit_tools(database),
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
    # context 没声明就从数据库 engine 反查
    return str(get_database(runtime_context).dialect)


# ── 与 SQLDatabaseToolkit 对齐的四个工具 ───────────────────────────────

class _QueryInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")


class _SchemaInput(BaseModel):
    table_names: str = Field(
        ...,
        description=(
            "A comma-separated list of the table names for which to return the schema. "
            "Example input: 'table1, table2, table3'"
        ),
    )


class _ListTablesInput(BaseModel):
    tool_input: str = Field("", description="An empty string")


class _CheckerInput(BaseModel):
    query: str = Field(..., description="A detailed and SQL query to be checked.")


# 与 langchain_community.tools.sql_database.prompt.QUERY_CHECKER 一致
QUERY_CHECKER = """
{query}
Double check the {dialect} query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only.

SQL Query: """


def _make_toolkit_tools(database: SqlDatabase) -> tuple[BaseTool, ...]:
    def sql_db_query(query: str) -> str:
        return database.run_no_throw(query)

    def sql_db_schema(table_names: str) -> str:
        return database.get_table_info_no_throw(
            [name.strip() for name in table_names.split(",")]
        )

    def sql_db_list_tables(tool_input: str = "") -> str:
        return ", ".join(database.get_usable_table_names())

    def sql_db_query_checker(query: str) -> str:
        # 经全局模型代理调用：自动带审计 / telemetry 回调，热替换后跟随新模型
        response = model.invoke(QUERY_CHECKER.format(query=query, dialect=database.dialect))
        return str(getattr(response, "content", response))

    return (
        StructuredTool.from_function(
            func=sql_db_query,
            name="sql_db_query",
            description=(
                "Input to this tool is a detailed and correct SQL query, output is a result "
                "from the database. If the query is not correct, an error message will be "
                "returned. If an error is returned, rewrite the query, check the query, and "
                "try again. If you encounter an issue with Unknown column 'xxxx' in 'field "
                "list', use sql_db_schema to query the correct table fields."
            ),
            args_schema=_QueryInput,
        ),
        StructuredTool.from_function(
            func=sql_db_schema,
            name="sql_db_schema",
            description=(
                "Input to this tool is a comma-separated list of tables, output is the schema "
                "and sample rows for those tables. Be sure that the tables actually exist by "
                "calling sql_db_list_tables first! Example Input: table1, table2, table3"
            ),
            args_schema=_SchemaInput,
        ),
        StructuredTool.from_function(
            func=sql_db_list_tables,
            name="sql_db_list_tables",
            description=(
                "Input is an empty string, output is a comma-separated list of tables in the "
                "database."
            ),
            args_schema=_ListTablesInput,
        ),
        StructuredTool.from_function(
            func=sql_db_query_checker,
            name="sql_db_query_checker",
            description=(
                "Use this tool to double check if your query is correct before executing it. "
                "Always use this tool before executing a query with sql_db_query!"
            ),
            args_schema=_CheckerInput,
        ),
    )


def _make_get_schema_tool(database: SqlDatabase) -> BaseTool:
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


def _make_find_slow_sql_tool(database: SqlDatabase) -> BaseTool:
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
