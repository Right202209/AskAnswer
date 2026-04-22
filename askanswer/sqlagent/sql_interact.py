from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
import os
from sqlalchemy import  text
from AskAnswer.askanswer.load import model

engine = SQLDatabase.from_uri(os.getenv("WLANGGRAPH_POSTGRES_DSN"))

toolkit = SQLDatabaseToolkit(db=engine, llm=model)
# ListSQLDatabaseTool（列出表）
# InfoSQLDatabaseTool（查 schema）
# QuerySQLDatabaseTool（执行 SQL）
# QuerySQLCheckerTool（检查 SQL）

@tool
def get_schema(_) -> str:
    """
    获取数据库所有表结构（给 LLM 用于生成 SQL）
    """
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ))
        tables = [row[0] for row in result]

    return "数据库表：" + ",".join(tables)

@tool
def find_slow_sql()-> str:
    """
    找慢 SQL
    """
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC;"
        ))

    return result
#TODO: 查看完整线程SELECT * FROM information_schema.processlist;
# 查看连接SELECT * FROM pg_stat_activity;
# 慢查询（需开启 log）SELECT query, calls, total_exec_time FROM pg_stat_statements ORDER BY total_exec_time DESC;
# PostgreSQL 锁 SELECT * FROM pg_locks;
# 查阻塞关系
# PostgreSQL 当前数据库状态
# ...

tools = [
    *toolkit.get_tools(),
    get_schema,
    find_slow_sql
    ]
sql_db_q = toolkit

print(tools)
