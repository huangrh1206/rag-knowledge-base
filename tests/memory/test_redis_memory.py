from src.config import Settings

from src.memory import LongTermMemory, MemoryScope, RedisMemoryStore


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    def hset(self, name: str, key: str, value: str) -> None:
        self.hashes.setdefault(name, {})[key] = value

    def hgetall(self, name: str) -> dict[str, str]:
        return dict(self.hashes.get(name, {}))

    def hdel(self, name: str, key: str) -> None:
        self.hashes.get(name, {}).pop(key, None)

    def scan_iter(self, match: str):
        prefix = match.removesuffix("*")
        return (key for key in self.hashes if key.startswith(prefix))


def test_redis_memory_store_is_scoped_and_updates_by_key() -> None:
    store = RedisMemoryStore(
        client=FakeRedis(),
        url="redis://test",
        prefix="rag:memory",
    )
    memory = LongTermMemory(store)
    scope = MemoryScope("tenant", "user", "support")
    other = MemoryScope("other", "user", "support")

    first = memory.remember(scope, "prefers concise", key="style")
    updated = memory.remember(scope, "prefers detailed", key="style")
    memory.remember(other, "other tenant fact", key="fact")

    assert updated.memory_id == first.memory_id
    assert memory.recall(scope, "detailed")[0].content == "prefers detailed"
    assert memory.recall(scope, "other") == []


def test_redis_memory_store_purges_expired_items() -> None:
    client = FakeRedis()
    store = RedisMemoryStore(
        client=client,
        url="redis://test",
        prefix="rag:memory",
    )
    memory = LongTermMemory(store)
    scope = MemoryScope("tenant", "user")
    memory.remember(
        scope,
        "temporary",
        key="temporary",
        expires_at="2020-01-01T00:00:00+00:00",
    )

    assert memory.recall(scope) == []
    assert memory.purge_expired("2021-01-01T00:00:00+00:00") == 1
    assert client.hashes["rag:memory:tenant:user:default"] == {}


def test_redis_memory_store_uses_settings(monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("REDIS_URL", "redis://configured.example/2")
    monkeypatch.setenv("REDIS_PREFIX", "configured:memory")
    client = FakeRedis()

    settings = Settings.from_env()
    store = RedisMemoryStore(client=client, settings=settings)
    memory = LongTermMemory(store)
    memory.remember(MemoryScope("tenant", "user"), "configured", key="source")

    assert "configured:memory:tenant:user:default" in client.hashes
