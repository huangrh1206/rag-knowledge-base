import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.models import Chunk
from src.qdrant_vector_store import QdrantVectorStore
from src.store_protocol import SearchStore


def chunks() -> list[Chunk]:
    return [
        Chunk(
            id="guide-0000",
            text="FastAPI",
            source="guide.docx",
            paragraph_start=1,
            paragraph_end=1,
        ),
        Chunk(
            id="guide-0001",
            text="Docker",
            source="guide.docx",
            paragraph_start=2,
            paragraph_end=2,
        ),
    ]


def test_qdrant_store_returns_highest_score_first() -> None:
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore.create(
        client=client,
        collection_name="test_chunks",
        chunks=chunks(),
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )

    search_store: SearchStore = store
    results = search_store.search(
        np.array([0.9, 0.1], dtype=np.float32),
        top_k=2,
    )

    assert [result.chunk.id for result in results] == [
        "guide-0000",
        "guide-0001",
    ]
    assert results[0].score > results[1].score
    assert results[0].chunk.source == "guide.docx"


def test_qdrant_store_rejects_count_mismatch() -> None:
    client = QdrantClient(location=":memory:")

    with pytest.raises(ValueError, match="same number"):
        QdrantVectorStore.create(
            client=client,
            collection_name="test_chunks",
            chunks=chunks(),
            embeddings=np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            ),
        )


def test_qdrant_store_rejects_query_dimension_mismatch() -> None:
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore.create(
        client=client,
        collection_name="test_chunks",
        chunks=chunks(),
        embeddings=np.eye(2, dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match="query dimension must match",
    ):
        store.search(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            top_k=1,
        )