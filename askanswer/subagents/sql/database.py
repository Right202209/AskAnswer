"""Native SQLAlchemy database access for the SQL subagent.

Replaces ``langchain_community.utilities.SQLDatabase`` + ``SQLDatabaseToolkit``
(``langchain-community`` is being sunset).  The tool **names, argument schemas and
output formats are kept identical** because the SQL subgraph and its prompts refer
to them by name:

- ``sql_db_list_tables(tool_input="")`` → ``"t1, t2"``
- ``sql_db_schema(table_names="t1, t2")`` → ``CREATE TABLE`` DDL + 3 sample rows per
  table, or ``"Error: ..."`` for unknown tables
- ``sql_db_query(query)`` → ``str(list[tuple])`` (``""`` for no rows), or
  ``"Error: ..."`` on SQLAlchemy errors so the model can rewrite the query
- ``sql_db_query_checker(query)`` → LLM double-check of the query (through the
  audited model proxy)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.types import NullType

# 与 langchain_community.SQLDatabase 默认值一致
DEFAULT_SAMPLE_ROWS = 3
DEFAULT_MAX_STRING_LENGTH = 300


def truncate_word(content: Any, *, length: int, suffix: str = "...") -> Any:
    """Truncate a string value at a word boundary so it fits ``length`` characters."""
    if not isinstance(content, str) or length <= 0:
        return content
    if len(content) <= length:
        return content
    return content[: length - len(suffix)].rsplit(" ", 1)[0] + suffix


class SqlDatabase:
    """Thin wrapper over a SQLAlchemy :class:`~sqlalchemy.engine.Engine`.

    Table names are inspected live on every call (new tables show up without
    rebuilding the wrapper).  Views are not listed, matching the previous
    ``SQLDatabase(view_support=False)`` default.
    """

    def __init__(
        self,
        engine: Engine,
        *,
        sample_rows_in_table_info: int = DEFAULT_SAMPLE_ROWS,
        max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
    ):
        self._engine = engine
        self._sample_rows = int(sample_rows_in_table_info)
        self._max_string_length = int(max_string_length)

    @classmethod
    def from_uri(cls, uri: str, **engine_kwargs: Any) -> "SqlDatabase":
        return cls(create_engine(uri, **engine_kwargs))

    @property
    def engine(self) -> Engine:
        return self._engine

    @property
    def dialect(self) -> str:
        return self._engine.dialect.name

    # ── schema ───────────────────────────────────────────────────────

    def get_usable_table_names(self) -> list[str]:
        names = inspect(self._engine).get_table_names()
        return sorted(name for name in names if not self._is_internal(name))

    def get_table_info(self, table_names: Sequence[str] | None = None) -> str:
        """``CREATE TABLE`` DDL plus sample rows for each table, sorted by DDL text."""
        usable = self.get_usable_table_names()
        if table_names is not None:
            missing = set(table_names).difference(usable)
            if missing:
                raise ValueError(f"table_names {missing} not found in database")
            usable = list(table_names)
        if not usable:
            return ""
        metadata = MetaData()
        metadata.reflect(bind=self._engine, only=list(usable))
        blocks = []
        for table in metadata.sorted_tables:
            if table.name not in usable:
                continue
            blocks.append(self._describe_table(table))
        blocks.sort()
        return "\n\n".join(blocks)

    def get_table_info_no_throw(self, table_names: Sequence[str] | None = None) -> str:
        try:
            return self.get_table_info(table_names)
        except ValueError as exc:
            return f"Error: {exc}"

    def _describe_table(self, table: Table) -> str:
        # 方言无法映射的列（NullType，如部分 JSON 类型）无法编译进 DDL，跳过
        for column in list(table.columns):
            if type(column.type) is NullType:
                table._columns.remove(column)
        info = str(CreateTable(table).compile(self._engine)).rstrip()
        if self._sample_rows > 0:
            info += f"\n\n/*\n{self._sample_rows_text(table)}\n*/"
        return info

    def _sample_rows_text(self, table: Table) -> str:
        columns = "\t".join(col.name for col in table.columns)
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(select(table).limit(self._sample_rows))
                sample = ["\t".join(str(value)[:100] for value in row) for row in rows]
        except ProgrammingError:
            # 部分方言在空表上会抛 ProgrammingError
            sample = []
        return f"{self._sample_rows} rows from {table.name} table:\n{columns}\n" + "\n".join(sample)

    # ── execution ────────────────────────────────────────────────────

    def execute(
        self, command: str, fetch: Literal["all", "one"] = "all"
    ) -> list[dict[str, Any]]:
        """Run one statement inside a transaction and return rows as dicts."""
        with self._engine.begin() as connection:
            cursor = connection.execute(text(command))
            if not cursor.returns_rows:
                return []
            if fetch == "one":
                first = cursor.fetchone()
                return [] if first is None else [first._asdict()]
            return [row._asdict() for row in cursor.fetchall()]

    def run(self, command: str, fetch: Literal["all", "one"] = "all") -> str:
        """Execute and stringify like ``SQLDatabase.run``: ``str(list[tuple])`` or ``""``."""
        rows = [
            tuple(
                truncate_word(value, length=self._max_string_length)
                for value in row.values()
            )
            for row in self.execute(command, fetch)
        ]
        return str(rows) if rows else ""

    def run_no_throw(self, command: str, fetch: Literal["all", "one"] = "all") -> str:
        try:
            return self.run(command, fetch)
        except SQLAlchemyError as exc:
            return f"Error: {exc}"

    def dispose(self) -> None:
        self._engine.dispose()

    def _is_internal(self, name: str) -> bool:
        return self.dialect == "sqlite" and name.startswith("sqlite_")
