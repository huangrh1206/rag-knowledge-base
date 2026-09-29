from pathlib import Path

from src.config import Settings
from src.memory import (
    InMemoryMemoryStore,
    SQLiteMemoryStore,
    create_memory_store,
)


def settings_for(**overrides) -> Settings:
    values = {
        "api_key": "test-key",
        "base_url": None,
        "chat_model": "chat",
        "embedding_model": "embedding",
    }
    values.update(overrides)
    return Settings(**values)


def test_factory_creates_in_memory_backend() -> None:
    store = create_memory_store(settings_for(memory_backend="memory"))
    assert isinstance(store, InMemoryMemoryStore)


def test_factory_creates_sqlite_backend(tmp_path) -> None:
    database = tmp_path / "memory.sqlite3"
    store = create_memory_store(
        settings_for(
            memory_backend="sqlite",
            memory_sqlite_path=Path(database),
        )
    )
    assert isinstance(store, SQLiteMemoryStore)
    store.close()
