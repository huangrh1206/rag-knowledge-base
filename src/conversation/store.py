"""Durable storage for short-term conversation snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Protocol

from src.memory.models import MemoryScope


class ConversationStore(Protocol):
    def load(self, scope: MemoryScope, session_id: str) -> dict[str, Any] | None: ...

    def save(
        self, scope: MemoryScope, session_id: str, snapshot: dict[str, Any]
    ) -> None: ...


class SQLiteConversationStore:
    """Store conversation snapshots in the configured SQLite database."""

    def __init__(self, database: str | Path) -> None:
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path))
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    snapshot TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tenant_id, user_id, scene_id, session_id)
                )
                """
            )

    def load(self, scope: MemoryScope, session_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT snapshot FROM conversations WHERE tenant_id = ? AND user_id = ? "
            "AND scene_id = ? AND session_id = ?",
            (scope.tenant_id, scope.user_id, scope.scene_id, session_id),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save(
        self, scope: MemoryScope, session_id: str, snapshot: dict[str, Any]
    ) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO conversations (
                    tenant_id, user_id, scene_id, session_id, snapshot, updated_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (tenant_id, user_id, scene_id, session_id)
                DO UPDATE SET snapshot = excluded.snapshot,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    scope.tenant_id,
                    scope.user_id,
                    scope.scene_id,
                    session_id,
                    json.dumps(snapshot, ensure_ascii=False),
                ),
            )

    def close(self) -> None:
        self._connection.close()
