from pathlib import Path

from src.api_client import create_openai_client
from src.config import Settings


def test_openai_client_uses_timeout_and_retries() -> None:
    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat-model",
        embedding_model="embedding-model",
        request_timeout=12.5,
        max_retries=3,
    )

    client = create_openai_client(settings)

    assert client.timeout == 12.5
    assert client.max_retries == 3
