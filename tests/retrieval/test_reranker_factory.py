from src.config import Settings
from src.retrieval.reranker import DisabledReranker
from src.retrieval.reranker_factory import reranker


def test_reranker_is_disabled_by_default() -> None:
    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat-model",
        embedding_model="embedding-model",
        rerank_enabled=False,
    )

    created_reranker = reranker(settings)

    assert isinstance(created_reranker, DisabledReranker)

def test_reranker_falls_back_when_model_unavailable(
    monkeypatch,
    caplog,
) -> None:
    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat-model",
        embedding_model="embedding-model",
        rerank_enabled=True,
    )

    def fail_loading(model_name):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        "src.retrieval.reranker_factory.CrossEncoderReranker",
        fail_loading,
    )

    with caplog.at_level(
        "WARNING",
        logger="src.retrieval.reranker_factory",
    ):
        created_reranker = reranker(settings)

    assert isinstance(created_reranker, DisabledReranker)
    assert any(
        "fallback to disabled reranker"
        in record.getMessage()
        for record in caplog.records
    )
