"""Persistence contracts and implementations for long-term memory."""

from collections.abc import Iterable
import json
from pathlib import Path
import sqlite3
from typing import Protocol

from src.memory.models import MemoryItem, MemoryScope, utc_now


class MemoryStore(Protocol):
    def upsert(self, item: MemoryItem) -> MemoryItem: 
        ...

    def search(
        self,
        scope: MemoryScope,
        query: str = "",
        *,
        memory_type: str | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]: 
        ...

    def delete(self, scope: MemoryScope, memory_id: str) -> None: 
        ...


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    def upsert(self, item: MemoryItem) -> MemoryItem:
        for existing in self._items.values():
            if (
                existing.scope == item.scope
                and existing.memory_type == item.memory_type
                and existing.key == item.key
            ):
                item.memory_id = existing.memory_id
                item.created_at = existing.created_at
                break
        existing = self._items.get(item.memory_id)
        if existing is not None:
            item.created_at = existing.created_at
        item.updated_at = utc_now()
        self._items[item.memory_id] = item
        return item

    def search(
        self,
        scope: MemoryScope,
        query: str = "",
        *,
        memory_type: str | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        if limit <= 0:
            return []
        terms = {term.lower() for term in query.split() if term}
        candidates = [
            item
            for item in self._items.values()
            if item.scope == scope
            and (memory_type is None or item.memory_type == memory_type)
        ]
        if terms:
            candidates = [
                item
                for item in candidates
                if terms & set(item.content.lower().split())
            ]
        candidates.sort(
            key=lambda item: (
                len(terms & set(item.content.lower().split())) if terms else 0,
                item.importance,
                item.updated_at,
            ),
            reverse=True,
        )
        return candidates[:limit]

    def delete(self, scope: MemoryScope, memory_id: str) -> None:
        item = self._items.get(memory_id)
        if item is not None and item.scope == scope:
            del self._items[memory_id]


class SQLiteMemoryStore:
    """SQLite store with tenant/user/scene filtering on every operation."""

    def __init__(self, database: str | Path) -> None:
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    memory_key TEXT,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(tenant_id, user_id, scene_id, memory_type, memory_key)
                )
                """
            )

    def upsert(self, item: MemoryItem) -> MemoryItem:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, user_id, scene_id, memory_type, memory_key)
                DO UPDATE SET content=excluded.content,
                    importance=excluded.importance,
                    metadata=excluded.metadata,
                    updated_at=excluded.updated_at
                """,
                (
                    item.memory_id,
                    item.scope.tenant_id,
                    item.scope.user_id,
                    item.scope.scene_id,
                    item.memory_type,
                    item.key,
                    item.content,
                    item.importance,
                    json.dumps(item.metadata, ensure_ascii=False),
                    item.created_at,
                    item.updated_at,
                ),
            )
        row = self._connection.execute(
            "SELECT * FROM memories WHERE tenant_id = ? AND user_id = ? "
            "AND scene_id = ? AND memory_type = ? AND memory_key IS ?",
            (
                item.scope.tenant_id,
                item.scope.user_id,
                item.scope.scene_id,
                item.memory_type,
                item.key,
            ),
        ).fetchone()
        return self._from_row(row) if row is not None else item

    def search(
        self,
        scope: MemoryScope,
        query: str = "",
        *,
        memory_type: str | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        if limit <= 0:
            return []
        clauses = ["tenant_id = ?", "user_id = ?", "scene_id = ?"]
        params: list[object] = [scope.tenant_id, scope.user_id, scope.scene_id]
        if memory_type is not None:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        rows = self._connection.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)} "
            "ORDER BY importance DESC, updated_at DESC LIMIT ?",
            [*params, limit * 5],
        ).fetchall()
        terms = {term.lower() for term in query.split() if term}
        items = [self._from_row(row) for row in rows]
        if terms:
            items.sort(
                key=lambda item: len(terms & set(item.content.lower().split())),
                reverse=True,
            )
        return items[:limit]

    def delete(self, scope: MemoryScope, memory_id: str) -> None:
        with self._connection:
            self._connection.execute(
                "DELETE FROM memories WHERE memory_id = ? AND tenant_id = ? "
                "AND user_id = ? AND scene_id = ?",
                (memory_id, scope.tenant_id, scope.user_id, scope.scene_id),
            )

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            memory_id=row["memory_id"],
            content=row["content"],
            memory_type=row["memory_type"],
            scope=MemoryScope(row["tenant_id"], row["user_id"], row["scene_id"]),
            key=row["memory_key"],
            importance=row["importance"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
