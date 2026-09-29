"""Factory for selecting the configured long-term memory backend."""

from src.config import Settings
from src.memory.store import (
    InMemoryMemoryStore,
    MemoryStore,
    RedisMemoryStore,
    SQLiteMemoryStore,
)


def create_memory_store(settings: Settings | None = None) -> MemoryStore:
    settings = settings or Settings.from_env()
    if settings.memory_backend == "memory":
        return InMemoryMemoryStore()
    if settings.memory_backend == "sqlite":
        settings.memory_sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        return SQLiteMemoryStore(settings.memory_sqlite_path)
    if settings.memory_backend == "redis":
        return RedisMemoryStore(
            url=settings.redis_url or None,
            prefix=settings.redis_prefix,
        )
    raise ValueError(f"unsupported memory backend: {settings.memory_backend}")
