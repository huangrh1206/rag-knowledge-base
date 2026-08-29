from pathlib import Path

import pytest

from src.config import Settings


def test_settings_uses_documented_model_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.delenv("RAG_CHAT_MODEL", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.chat_model == "gpt-4.1-mini"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_batch_size == 20
    assert settings.evidence_minimum_score == 0.50
    assert settings.evidence_minimum_results == 1
    assert settings.vector_store_backend == "numpy"
    assert settings.qdrant_path == Path("storage/qdrant")
    assert settings.qdrant_collection == "rag_chunks"

def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("RAG_CHAT_MODEL", "chat-model")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "12")
    monkeypatch.setenv("RAG_EVIDENCE_MIN_SCORE", "0.75")
    monkeypatch.setenv("RAG_EVIDENCE_MIN_RESULTS", "2")
    monkeypatch.setenv("RAG_VECTOR_STORE_BACKEND", "qdrant")
    monkeypatch.setenv("RAG_QDRANT_PATH", "custom/qdrant")
    monkeypatch.setenv("RAG_QDRANT_COLLECTION", "custom_chunks")
    settings = Settings.from_env()

    assert settings.api_key == "test-key"
    assert settings.chat_model == "chat-model"
    assert settings.embedding_model == "embedding-model"
    assert settings.top_k == 3
    assert settings.embedding_batch_size == 12
    assert settings.evidence_minimum_score == 0.75
    assert settings.evidence_minimum_results == 2
    assert settings.index_dir == Path("storage/index")
    assert settings.vector_store_backend == "qdrant"
    assert settings.qdrant_path == Path("custom/qdrant")
    assert settings.qdrant_collection == "custom_chunks"


def test_settings_reject_overlap_not_smaller_than_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")

    with pytest.raises(ValueError, match="overlap"):
        Settings.from_env()


def test_settings_reject_invalid_evidence_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("RAG_EVIDENCE_MIN_SCORE", "1.5")

    with pytest.raises(ValueError, match="evidence minimum score"):
        Settings.from_env()

def test_settings_reject_invalid_vector_store_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("RAG_VECTOR_STORE_BACKEND", "unknown")

    with pytest.raises(
        ValueError,
        match="vector store backend",
    ):
        Settings.from_env()

def test_settings_reject_non_positive_timeout(monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("RAG_REQUEST_TIMEOUT", "0")

    with pytest.raises(ValueError, match="request timeout"):
        Settings.from_env()


def test_settings_reject_negative_retries(monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("RAG_MAX_RETRIES", "-1")

    with pytest.raises(ValueError, match="max retries"):
        Settings.from_env()