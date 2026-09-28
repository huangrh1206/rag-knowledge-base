from src.memory import (
    InMemoryMemoryStore,
    LongTermMemory,
    MemoryScope,
    ShortTermMemory,
    RuleBasedMemoryExtractor,
)


def test_long_term_memory_is_scoped_and_searchable() -> None:
    store = InMemoryMemoryStore()
    memory = LongTermMemory(store)
    scope = MemoryScope("tenant-a", "user-a", "support")
    other = MemoryScope("tenant-b", "user-a", "support")
    item = memory.remember(scope, "User prefers concise answers", key="style")
    memory.remember(other, "Other tenant fact", key="fact")

    assert memory.recall(scope, "concise")[0].memory_id == item.memory_id
    assert memory.recall(scope, "Other") == []


def test_short_term_memory_preserves_system_and_recent_messages() -> None:
    memory = ShortTermMemory(max_messages=3, max_chars=100)
    memory.extend([
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "latest"},
    ])

    assert memory.messages() == [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "latest"},
    ]


def test_rule_extractor_updates_same_preference_key() -> None:
    store = InMemoryMemoryStore()
    memory = LongTermMemory(store)
    scope = MemoryScope("tenant", "user")
    extractor = RuleBasedMemoryExtractor()
    memory.extract_and_remember(
        scope,
        [{"role": "user", "content": "I prefer concise answers"}],
        extractor,
    )
    memory.extract_and_remember(
        scope,
        [{"role": "user", "content": "I prefer detailed answers"}],
        extractor,
    )

    items = memory.recall(scope, memory_type="preference")
    assert len(items) == 1
    assert items[0].content == "I prefer detailed answers"
