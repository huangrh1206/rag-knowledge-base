from src.task.agent import GeneralTaskAgent

from src.conversation import ConversationContext
from src.memory import InMemoryMemoryStore, LongTermMemory, MemoryScope, RuleBasedMemoryExtractor


def test_general_agent_chat_persists_and_recalls_long_term_memory():
    scope = MemoryScope("tenant", "user", "general")
    context = ConversationContext(
        scope,
        long_term=LongTermMemory(InMemoryMemoryStore()),
        extractor=RuleBasedMemoryExtractor(),
    )
    answers = iter(["已记录", "好的，我会保持简洁。"])
    agent = GeneralTaskAgent(responder=lambda *_: next(answers), conversation=context)

    agent.chat("我偏好简洁回答")
    agent.chat("以后怎么回答？")

    assert context.long_term.recall(scope, memory_type="preference")
    assert any("简洁回答" in str(item.get("content")) for item in context.build_messages("回答风格"))


def test_general_agent_reset_clears_short_term_but_keeps_long_term_memory():
    scope = MemoryScope("tenant", "user", "general")
    context = ConversationContext(
        scope,
        long_term=LongTermMemory(InMemoryMemoryStore()),
        extractor=RuleBasedMemoryExtractor(),
    )
    agent = GeneralTaskAgent(responder=lambda *_: "收到", conversation=context)
    agent.chat("我喜欢 Markdown")
    agent.reset_chat()

    assert agent.messages == []
    assert context.long_term.recall(scope, memory_type="preference")

def test_general_agent_has_no_retriever_contract():
    agent = GeneralTaskAgent()
    assert not hasattr(agent, "retriever")

def test_general_agent_chat_retains_history():
    agent = GeneralTaskAgent()
    first = agent.chat("第一轮")
    second = agent.chat("第二轮")
    assert first.status.value == "completed"
    assert second.status.value == "completed"
    assert [item["role"] for item in agent.messages] == ["user", "assistant", "user", "assistant"]
    assert "第一轮" in agent.messages[2]["content"] or "第一轮" in str(second.final_answer)
