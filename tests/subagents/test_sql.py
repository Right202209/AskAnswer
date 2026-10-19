"""SQL 子 agent：只读闸门、按 (tenant, DSN) 隔离的连接缓存、原生 SQLAlchemy 工具与原 toolkit 同名同语义。

全部跑在临时 SQLite 文件上，不需要真实数据库或 API key。
"""

from __future__ import annotations

import sqlite3

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from askanswer.config.schema import ContextSchema
from askanswer.subagents.sql import sql_interact, sql_node
from askanswer.subagents.sql.sql_node import _check_sql_readonly


@pytest.fixture
def sqlite_dsn(tmp_path):
    db = tmp_path / "shop.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);
        INSERT INTO users(name) VALUES ('alice'), ('bob');
        INSERT INTO orders(user_id, amount) VALUES (1, 9.5);
        """
    )
    conn.commit()
    conn.close()
    sql_interact.clear_sql_caches()
    yield f"sqlite:///{db}"
    sql_interact.clear_sql_caches()


# ── 只读闸门 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM users",
        "select count(*) from orders where amount > 1",
        "WITH t AS (SELECT 1 AS x) SELECT * FROM t",
        "-- comment\nSELECT 1",
        "/* block */ SELECT 1; SELECT 2",
        "",
    ],
)
def test_readonly_gate_allows_queries(sql):
    assert _check_sql_readonly(sql) is None


@pytest.mark.parametrize(
    "sql,keyword",
    [
        ("DELETE FROM users", "delete"),
        ("update users set name='x'", "update"),
        ("INSERT INTO users(name) VALUES ('x')", "insert"),
        ("DROP TABLE users", "drop"),
        ("ALTER TABLE users ADD COLUMN x INT", "alter"),
        ("truncate orders", "truncate"),
        ("SELECT 1; DELETE FROM users", "delete"),
        ("/* hide */ DROP TABLE users", "drop"),
        ("WITH t AS (SELECT 1) DELETE FROM users", "delete"),
        ("ATTACH DATABASE 'x.db' AS x", "attach"),
    ],
)
def test_readonly_gate_blocks_dml_ddl(sql, keyword):
    assert _check_sql_readonly(sql) == keyword


class _Runtime:
    def __init__(self, context):
        self.context = context


def test_run_query_node_refuses_dml_without_executing(sqlite_dsn):
    runtime = _Runtime(ContextSchema(db_dsn=sqlite_dsn))
    call = {"name": "sql_db_query", "args": {"query": "DELETE FROM users"}, "id": "c1"}
    out = sql_node.run_query_node({"messages": [AIMessage(content="", tool_calls=[call])]}, runtime)
    (message,) = out["messages"]
    assert isinstance(message, ToolMessage)
    assert "DELETE" in message.content and "只读" in message.content
    # 数据未被删除
    rows = sql_interact.get_sql_tool("sql_db_query", runtime.context).invoke(
        {"query": "SELECT count(*) FROM users"}
    )
    assert rows == "[(2,)]"


def test_run_query_node_executes_select_and_counts(sqlite_dsn):
    runtime = _Runtime(ContextSchema(db_dsn=sqlite_dsn))
    call = {"name": "sql_db_query", "args": {"query": "SELECT name FROM users ORDER BY id"},
            "id": "c1"}
    out = sql_node.run_query_node(
        {"messages": [AIMessage(content="", tool_calls=[call])], "query_count": 1}, runtime
    )
    assert out["messages"][0].content == "[('alice',), ('bob',)]"
    assert out["query_count"] == 2


# ── 连接缓存的租户隔离 ───────────────────────────────────────────────────

def test_cache_isolated_per_tenant_for_same_dsn(sqlite_dsn):
    alice = ContextSchema(db_dsn=sqlite_dsn, tenant_id="alice")
    bob = ContextSchema(db_dsn=sqlite_dsn, tenant_id="bob")
    db_alice = sql_interact.get_database(alice)
    db_bob = sql_interact.get_database(bob)
    assert db_alice is not db_bob
    assert db_alice.engine is not db_bob.engine  # 不共享连接池
    assert sql_interact.get_sql_tools(alice) is not sql_interact.get_sql_tools(bob)


def test_cache_reused_within_same_tenant(sqlite_dsn):
    ctx = ContextSchema(db_dsn=sqlite_dsn, tenant_id="alice")
    assert sql_interact.get_database(ctx) is sql_interact.get_database(dict(ctx.__dict__))
    assert sql_interact.get_sql_tools(ctx) is sql_interact.get_sql_tools(ctx)


def test_missing_dsn_raises(monkeypatch):
    monkeypatch.delenv("ASKANSWER_DB_DSN", raising=False)
    with pytest.raises(RuntimeError):
        sql_interact.get_database_uri(ContextSchema())


# ── 原生 SQLAlchemy 工具：名字 / 输出格式与原 SQLDatabaseToolkit 一致 ─────────

def test_tool_names_preserved(sqlite_dsn):
    names = [t.name for t in sql_interact.get_sql_tools(ContextSchema(db_dsn=sqlite_dsn))]
    assert names == [
        "sql_db_query",
        "sql_db_schema",
        "sql_db_list_tables",
        "sql_db_query_checker",
        "get_schema",
        "find_slow_sql",
    ]


def test_list_tables_and_dialect(sqlite_dsn):
    ctx = ContextSchema(db_dsn=sqlite_dsn)
    assert sql_interact.get_sql_tool("sql_db_list_tables", ctx).invoke({}) == "orders, users"
    assert sql_interact.get_sql_dialect(ctx) == "sqlite"
    assert sql_interact.get_sql_dialect(ContextSchema(db_dsn=sqlite_dsn, db_dialect="pg")) == "pg"


def test_schema_tool_ddl_and_sample_rows(sqlite_dsn):
    schema = sql_interact.get_sql_tool("sql_db_schema", ContextSchema(db_dsn=sqlite_dsn)).invoke(
        {"table_names": "users"}
    )
    assert "CREATE TABLE users" in schema
    assert "3 rows from users table:" in schema
    assert "alice" in schema


def test_schema_tool_unknown_table_returns_error_text(sqlite_dsn):
    schema = sql_interact.get_sql_tool("sql_db_schema", ContextSchema(db_dsn=sqlite_dsn)).invoke(
        {"table_names": "nope"}
    )
    assert schema.startswith("Error: table_names {'nope'} not found")


def test_query_tool_formats_and_errors(sqlite_dsn):
    tool = sql_interact.get_sql_tool("sql_db_query", ContextSchema(db_dsn=sqlite_dsn))
    assert tool.invoke({"query": "SELECT id, amount FROM orders"}) == "[(1, 9.5)]"
    assert tool.invoke({"query": "SELECT * FROM users WHERE 1 = 0"}) == ""
    assert tool.invoke({"query": "SELECT * FROM nope"}).startswith("Error: ")


def test_find_slow_sql_non_postgres(sqlite_dsn):
    tool = sql_interact.get_sql_tool("find_slow_sql", ContextSchema(db_dsn=sqlite_dsn))
    assert tool.invoke({}) == "find_slow_sql 仅支持 PostgreSQL。"


def test_query_checker_uses_model(fake_llm, sqlite_dsn):
    fake_llm.responses = ["SELECT name FROM users LIMIT 5"]
    tool = sql_interact.get_sql_tool("sql_db_query_checker", ContextSchema(db_dsn=sqlite_dsn))
    assert tool.invoke({"query": "SELECT name FROM users"}) == "SELECT name FROM users LIMIT 5"
    prompt = fake_llm.calls[-1][0].content
    assert "Double check the sqlite query" in prompt


def test_no_langchain_community_import():
    import sys

    import askanswer.subagents.sql.sql_interact  # noqa: F401

    assert not any(name.startswith("langchain_community") for name in sys.modules)


# ── DSN 环境变量：ASKANSWER_DB_DSN 为主，WLANGGRAPH_POSTGRES_DSN 废弃回退 ─────────

def test_dsn_from_askanswer_env(monkeypatch, sqlite_dsn):
    monkeypatch.setenv("ASKANSWER_DB_DSN", sqlite_dsn)
    monkeypatch.setenv("WLANGGRAPH_POSTGRES_DSN", "postgresql://legacy/ignored")
    assert sql_interact.get_database_uri(ContextSchema()) == sqlite_dsn


def test_legacy_dsn_env_warns(monkeypatch, sqlite_dsn):
    monkeypatch.setenv("WLANGGRAPH_POSTGRES_DSN", sqlite_dsn)
    with pytest.deprecated_call():
        assert sql_interact.get_database_uri(ContextSchema()) == sqlite_dsn


def test_context_dsn_beats_env(monkeypatch, sqlite_dsn):
    monkeypatch.setenv("ASKANSWER_DB_DSN", "postgresql://env/ignored")
    assert sql_interact.get_database_uri(ContextSchema(db_dsn=sqlite_dsn)) == sqlite_dsn


def test_runtime_context_from_env_reads_new_dsn(monkeypatch):
    from askanswer.core.runner import runtime_context_from_env

    monkeypatch.setenv("ASKANSWER_DB_DSN", "sqlite:///x.db")
    monkeypatch.setenv("ASKANSWER_DB_DIALECT", "sqlite")
    ctx = runtime_context_from_env()
    assert (ctx.db_dsn, ctx.db_dialect) == ("sqlite:///x.db", "sqlite")


def test_runtime_context_from_env_legacy_dsn_warns(monkeypatch):
    from askanswer.core.runner import runtime_context_from_env

    monkeypatch.setenv("WLANGGRAPH_POSTGRES_DSN", "postgresql://legacy/db")
    with pytest.deprecated_call():
        assert runtime_context_from_env().db_dsn == "postgresql://legacy/db"


def test_sql_clarify_prompt_mentions_new_env():
    from askanswer.intents import get_intent_registry

    request = get_intent_registry().get("sql").clarify({}, ContextSchema())
    assert "ASKANSWER_DB_DSN" in request.prompt
