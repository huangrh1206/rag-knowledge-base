import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.config import Settings
from src.models import Chunk
from src.qdrant_vector_store import QdrantVectorStore
from src.store_factory import load_search_store
from src.vector_store import VectorStore


def test_factory_loads_numpy_store(tmp_path) -> None:
    index_dir = tmp_path / "index"

    VectorStore(
        [
            Chunk(
                id="guide-0000",
                text="content",
                source="guide.docx",
                paragraph_start=1,
                paragraph_end=1,
            )
        ],
        np.array([[1.0, 0.0]], dtype=np.float32),
    ).save(index_dir)

    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat-model",
        embedding_model="embedding-model",
        index_dir=index_dir,
        vector_store_backend="numpy",
    )

    store = load_search_store(settings)

    results = store.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert results[0].chunk.id == "guide-0000"


def test_factory_loads_qdrant_store(tmp_path) -> None:
    qdrant_path = tmp_path / "qdrant"
    client = QdrantClient(path=str(qdrant_path))

    QdrantVectorStore.create(
        client=client,
        collection_name="rag_chunks",
        chunks=[
            Chunk(
                id="guide-0000",
                text="content",
                source="guide.docx",
                paragraph_start=1,
                paragraph_end=1,
            )
        ],
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat-model",
        embedding_model="embedding-model",
        vector_store_backend="qdrant",
        qdrant_path=qdrant_path,
        qdrant_collection="rag_chunks",
    )

    store = load_search_store(
        settings=settings,
        qdrant_client=client,
    )

    results = store.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert results[0].chunk.id == "guide-0000"
    client.close()


def test_factory_rejects_unknown_backend() -> None:
    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat-model",
        embedding_model="embedding-model",
        vector_store_backend="invalid",
    )

    with pytest.raises(
        ValueError,
        match="unsupported vector store backend",
    ):
        load_search_store(settings)