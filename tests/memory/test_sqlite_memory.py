from src.memory import LongTermMemory, MemoryScope, SQLiteMemoryStore


def test_sqlite_memory_persists_and_isolates_scope(tmp_path) -> None:
    database = tmp_path / "memory.sqlite3"
    scope = MemoryScope("tenant", "user", "scene")
    store = SQLiteMemoryStore(database)
    memory = LongTermMemory(store)
    item = memory.remember(scope, "prefers markdown", key="format")
    store.close()

    reopened = SQLiteMemoryStore(database)
    assert reopened.search(scope, "markdown")[0].memory_id == item.memory_id
    assert reopened.search(MemoryScope("tenant", "other", "scene")) == []
    reopened.close()
