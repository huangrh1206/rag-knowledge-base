import pytest


RAG_ENV_KEYS = (
    "RAG_API_KEY",
    "RAG_BASE_URL",
    "RAG_CHAT_MODEL",
    "RAG_EMBEDDING_MODEL",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_TOP_K",
    "RAG_SIMILARITY_THRESHOLD",
    "RAG_INDEX_DIR",
)


@pytest.fixture(autouse=True)
def isolate_rag_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.config.load_dotenv", lambda: False)
    for key in RAG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
