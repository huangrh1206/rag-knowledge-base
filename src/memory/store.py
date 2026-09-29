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

    def purge_expired(self, now: str | None = None) -> int: 
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
            and not item.is_expired()
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

    def purge_expired(self, now: str | None = None) -> int:
        expired = [item.memory_id for item in self._items.values() if item.is_expired(now)]
        for memory_id in expired:
            del self._items[memory_id]
        return len(expired)


class RedisMemoryStore:
    """Redis-backed memory store using one hash per isolated memory scope.

    The Redis package is optional. A client can be injected for production or
    tests; otherwise ``redis.from_url`` is imported only when this class is
    instantiated.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        client=None,
        prefix: str | None = None,
        settings=None,
    ) -> None:
        if settings is not None:
            url = url or settings.redis_url
            prefix = prefix or settings.redis_prefix
        if client is not None:
            import os

            url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
            prefix = prefix or os.getenv("REDIS_PREFIX", "rag:memory")
        elif url is None or prefix is None:
            from src.config import Settings

            settings = Settings.from_env()
            url = url if url is not None else settings.redis_url
            prefix = prefix if prefix is not None else settings.redis_prefix
        if not url:
            raise ValueError("REDIS_URL is required when using RedisMemoryStore")
        if client is None:
            try:
                import redis
            except ImportError as exc:
                raise RuntimeError(
                    "RedisMemoryStore requires the optional 'redis' package"
                ) from exc
            client = redis.from_url(url, decode_responses=True)
        self._client = client
        self._prefix = prefix

    def upsert(self, item: MemoryItem) -> MemoryItem:
        key = self._scope_key(item.scope)
        existing = self._find_by_identity(key, item.memory_type, item.key)
        if existing is not None:
            item.memory_id = existing.memory_id
            item.created_at = existing.created_at
        item.updated_at = utc_now()
        self._client.hset(key, item.memory_id, json.dumps(self._to_dict(item), ensure_ascii=False))
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
        items = [
            item
            for item in self._all(scope)
            if (memory_type is None or item.memory_type == memory_type)
            and not item.is_expired()
        ]
        if terms:
            items = [
                item for item in items
                if terms & set(item.content.lower().split())
            ]
        items.sort(
            key=lambda item: (
                len(terms & set(item.content.lower().split())) if terms else 0,
                item.importance,
                item.updated_at,
            ),
            reverse=True,
        )
        return items[:limit]

    def delete(self, scope: MemoryScope, memory_id: str) -> None:
        self._client.hdel(self._scope_key(scope), memory_id)

    def purge_expired(self, now: str | None = None) -> int:
        removed = 0
        for key in self._client.scan_iter(match=f"{self._prefix}:*"):
            values = self._client.hgetall(key)
            for memory_id, payload in values.items():
                if self._from_json(payload).is_expired(now):
                    self._client.hdel(key, memory_id)
                    removed += 1
        return removed

    def _all(self, scope: MemoryScope) -> list[MemoryItem]:
        return [self._from_json(payload) for payload in self._client.hgetall(self._scope_key(scope)).values()]

    def _find_by_identity(self, scope_key: str, memory_type: str, memory_key: str | None) -> MemoryItem | None:
        for payload in self._client.hgetall(scope_key).values():
            item = self._from_json(payload)
            if item.memory_type == memory_type and item.key == memory_key:
                return item
        return None

    def _scope_key(self, scope: MemoryScope) -> str:
        return f"{self._prefix}:{scope.tenant_id}:{scope.user_id}:{scope.scene_id}"

    @staticmethod
    def _to_dict(item: MemoryItem) -> dict[str, object]:
        return {
            "memory_id": item.memory_id,
            "tenant_id": item.scope.tenant_id,
            "user_id": item.scope.user_id,
            "scene_id": item.scope.scene_id,
            "memory_type": item.memory_type,
            "key": item.key,
            "content": item.content,
            "importance": item.importance,
            "metadata": item.metadata,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "expires_at": item.expires_at,
        }

    @staticmethod
    def _from_json(payload: str | bytes) -> MemoryItem:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        data = json.loads(payload)
        return MemoryItem(
            memory_id=data["memory_id"],
            content=data["content"],
            memory_type=data["memory_type"],
            scope=MemoryScope(data["tenant_id"], data["user_id"], data["scene_id"]),
            key=data["key"],
            importance=data["importance"],
            metadata=data["metadata"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            expires_at=data.get("expires_at"),
        )


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
                    expires_at TEXT,
                    UNIQUE(tenant_id, user_id, scene_id, memory_type, memory_key)
                )
                """
            )
            columns = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(memories)"
                ).fetchall()
            }
            if "expires_at" not in columns:
                self._connection.execute("ALTER TABLE memories ADD COLUMN expires_at TEXT")

    def upsert(self, item: MemoryItem) -> MemoryItem:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO memories (
                    memory_id, tenant_id, user_id, scene_id, memory_type,
                    memory_key, content, importance, metadata, created_at,
                    updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, user_id, scene_id, memory_type, memory_key)
                DO UPDATE SET content=excluded.content,
                    importance=excluded.importance,
                    metadata=excluded.metadata,
                    expires_at=excluded.expires_at,
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
                    item.expires_at,
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
            "AND (expires_at IS NULL OR expires_at > ?) "
            "ORDER BY importance DESC, updated_at DESC LIMIT ?",
            [*params, utc_now(), limit * 5],
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

    def purge_expired(self, now: str | None = None) -> int:
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now or utc_now(),),
            )
        return cursor.rowcount

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
            expires_at=row["expires_at"],
        )
