import pytest

from src.conversation import ConversationContext, ConversationStatus
from src.memory import (
    InMemoryMemoryStore,
    LongTermMemory,
    MemoryScope,
    RuleBasedMemoryExtractor,
)


def test_conversation_context_combines_memory_and_turns() -> None:
    scope = MemoryScope("tenant", "user")
    long_term = LongTermMemory(InMemoryMemoryStore())
    long_term.remember(scope, "User prefers concise answers", key="style")
    context = ConversationContext(
        scope,
        long_term=long_term,
        extractor=RuleBasedMemoryExtractor(),
    )

    first = context.build_messages("How concise should you answer?")
    assert first[0]["role"] == "system"
    assert "concise" in first[0]["content"]
    assert first[-1] == {
        "role": "user",
        "content": "How concise should you answer?",
    }

    context.record(first + [{"role": "assistant", "content": "Briefly."}])
    second = context.build_messages("What about now?")
    assert second[-1] == {"role": "user", "content": "What about now?"}
    assert second[-2] == {"role": "assistant", "content": "Briefly."}

    context.record_turn("I prefer detailed answers", "Understood.")
    assert long_term.recall(scope, memory_type="preference")[0].content == (
        "I prefer detailed answers"
    )


def test_conversation_can_pause_resume_and_complete() -> None:
    context = ConversationContext(MemoryScope("tenant", "user"))
    context.record_turn("hello", "hi")
    assert context.status is ConversationStatus.ACTIVE
    assert context.turn_count == 1

    context.pause()
    with pytest.raises(RuntimeError, match="not active"):
        context.build_messages("continue")

    context.resume()
    assert context.build_messages("continue")[-1]["content"] == "continue"
    context.complete()
    with pytest.raises(RuntimeError, match="not active"):
        context.build_messages("after completion")

    context.reset()
    assert context.status is ConversationStatus.ACTIVE
    assert context.turn_count == 0


def test_conversation_snapshot_round_trips() -> None:
    context = ConversationContext(MemoryScope("tenant", "user"))
    context.record_turn("hello", "hi")
    context.pause()

    restored = ConversationContext(MemoryScope("tenant", "user"))
    restored.restore(context.snapshot())

    assert restored.status is ConversationStatus.PAUSED
    assert restored.turn_count == 1
    assert restored.short_term.messages()[-1]["content"] == "hi"


def test_conversation_from_settings_wires_memory_limits(tmp_path) -> None:
    from pathlib import Path
    from src.config import Settings

    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat",
        embedding_model="embedding",
        memory_backend="sqlite",
        memory_sqlite_path=Path(tmp_path) / "memory.sqlite3",
        memory_max_messages=4,
        memory_max_chars=1000,
        memory_max_tokens=200,
        memory_min_confidence=0.9,
    )
    context = ConversationContext.from_settings(
        MemoryScope("tenant", "user"),
        settings=settings,
    )

    assert context.short_term.max_messages == 4
    assert context.short_term.max_chars == 1000
    assert context.short_term.max_tokens == 200
    assert context.long_term is not None


def test_conversation_short_term_reopens_from_sqlite(tmp_path) -> None:
    from pathlib import Path
    from src.config import Settings

    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat",
        embedding_model="embedding",
        memory_backend="sqlite",
        memory_sqlite_path=Path(tmp_path) / "memory.sqlite3",
    )
    scope = MemoryScope("tenant", "user", "general")

    first = ConversationContext.from_settings(
        scope,
        settings=settings,
        session_id="session-1",
    )
    first.record_turn("我住在南京", "好的，我记住了")

    second = ConversationContext.from_settings(
        scope,
        settings=settings,
        session_id="session-1",
    )
    assert second.short_term.messages() == [
        {"role": "user", "content": "我住在南京"},
        {"role": "assistant", "content": "好的，我记住了"},
    ]

    isolated = ConversationContext.from_settings(
        scope,
        settings=settings,
        session_id="other-session",
    )
    assert isolated.short_term.messages() == []


def test_settings_context_extracts_long_term_memory_by_default(tmp_path) -> None:
    from pathlib import Path
    from src.config import Settings
    from src.memory.store import SQLiteMemoryStore

    database = Path(tmp_path) / "memory.sqlite3"
    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat",
        embedding_model="embedding",
        memory_backend="sqlite",
        memory_sqlite_path=database,
    )
    scope = MemoryScope("tenant", "user", "general")

    context = ConversationContext.from_settings(
        scope,
        settings=settings,
        session_id="session-1",
    )
    context.record_turn(
        "记住我在南京江宁区的河海大学，是河海大学的研三学生。我旅游喜欢博物馆。",
        "好的，我记住了。",
    )

    store = SQLiteMemoryStore(database)
    items = store.search(scope, "河海大学", limit=10)
    assert any("河海大学" in item.content for item in items)
    assert any(item.memory_type == "preference" for item in store.search(scope, "博物馆"))
    store.close()


def test_reopening_existing_session_backfills_long_term_memory(tmp_path) -> None:
    from pathlib import Path
    from src.config import Settings
    from src.conversation.store import SQLiteConversationStore
    from src.memory.store import SQLiteMemoryStore

    database = Path(tmp_path) / "memory.sqlite3"
    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat",
        embedding_model="embedding",
        memory_backend="sqlite",
        memory_sqlite_path=database,
    )
    scope = MemoryScope("tenant", "user", "general")
    SQLiteConversationStore(database).save(
        scope,
        "old-session",
        {
            "status": "active",
            "turn_count": 1,
            "messages": [
                {"role": "user", "content": "我在河海大学上学。"},
                {"role": "assistant", "content": "好的。"},
            ],
        },
    )

    ConversationContext.from_settings(
        scope,
        settings=settings,
        session_id="old-session",
    )

    store = SQLiteMemoryStore(database)
    assert any("河海大学" in item.content for item in store.search(scope, "河海大学"))
    store.close()
