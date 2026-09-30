from unittest.mock import Mock, patch

from src.api.factory import create_agent
from src.config import Settings


def test_create_agent_wires_client_retriever_and_conversation() -> None:
    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat-model",
        embedding_model="embedding-model",
        memory_backend="memory",
    )
    client = Mock()
    retriever = Mock()

    with patch("src.api.factory.create_openai_client", return_value=client), \
         patch("src.api.factory.create_retriever", return_value=retriever), \
         patch("src.api.factory.ConversationContext.from_settings") as context:
        agent = create_agent(settings)

    assert agent._retriever is retriever
    assert agent._conversation is None
    context.assert_not_called()
