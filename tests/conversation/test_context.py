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
