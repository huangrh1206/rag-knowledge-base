from pathlib import Path

from src.api_client import create_openai_client
from src.config import Settings


def test_openai_client_disables_sdk_retries() -> None:
    settings = Settings(
        api_key="test-key",
        base_url="https://example.test/v1",
        chat_model="chat-model",
        embedding_model="embedding-model",
        index_dir=Path("storage/index"),
    )

    client = create_openai_client(settings)

    assert client.max_retries == 0
