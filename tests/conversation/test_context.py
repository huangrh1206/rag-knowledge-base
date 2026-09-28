from src.conversation import ConversationContext
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
