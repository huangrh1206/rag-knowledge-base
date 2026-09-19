"""Factories for LangGraph checkpointer backends."""

from contextlib import AbstractContextManager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def open_sqlite_checkpointer(
    database: str | Path,
) -> AbstractContextManager[SqliteSaver]:
    """Open a SQLite checkpointer whose lifetime is managed by ``with``."""

    return SqliteSaver.from_conn_string(str(database))
