"""SQLite persistence for Harness sessions and audit events."""

import json
from pathlib import Path
import sqlite3

from src.harness.models import HarnessEvent, HarnessSession


class SQLiteSessionStore:
    def __init__(self, database: str | Path) -> None:
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        self._create_schema()

    def create(self, run_id: str) -> HarnessSession:
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO harness_sessions (run_id) VALUES (?)",
                    (run_id,),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"run already exists: {run_id}") from exc
        return HarnessSession(run_id=run_id)

    def get(self, run_id: str) -> HarnessSession:
        session = self._connection.execute(
            "SELECT run_id FROM harness_sessions WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if session is None:
            raise KeyError(f"run not found: {run_id}")

        event_rows = self._connection.execute(
            """
            SELECT event_type, payload, created_at
            FROM harness_events
            WHERE run_id = ?
            ORDER BY sequence_id
            """,
            (run_id,),
        ).fetchall()
        events = [
            HarnessEvent(
                run_id=run_id,
                event_type=row["event_type"],
                payload=json.loads(row["payload"]),
                created_at=row["created_at"],
            )
            for row in event_rows
        ]
        return HarnessSession(
            run_id=run_id,
            events=events,
        )

    def append(self, run_id: str, event: HarnessEvent) -> None:
        self._require_session(run_id)
        if event.run_id != run_id:
            raise ValueError("event run_id does not match target session")
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO harness_events (
                    run_id,
                    event_type,
                    payload,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    event.event_type,
                    json.dumps(event.payload, ensure_ascii=False),
                    event.created_at,
                ),
            )

    def close(self) -> None:
        self._connection.close()

    def _require_session(self, run_id: str) -> None:
        row = self._connection.execute(
            "SELECT 1 FROM harness_sessions WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"run not found: {run_id}")

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS harness_sessions (
                    run_id TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS harness_events (
                    sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES harness_sessions(run_id)
                );
                """
            )
