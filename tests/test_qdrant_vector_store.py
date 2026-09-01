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
    assert store.vector_dimension == 2


def test_qdrant_store_returns_all_chunks_for_lexical_retrieval() -> None:
    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore.create(
        client=client,
        collection_name="test_chunks",
        chunks=chunks(),
        embeddings=np.eye(2, dtype=np.float32),
    )

    loaded_chunks = store.all_chunks()

    assert {chunk.id for chunk in loaded_chunks} == {
        "guide-0000",
        "guide-0001",
    }
    assert {chunk.text for chunk in loaded_chunks} == {
        "FastAPI",
        "Docker",
    }


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

def test_qdrant_store_loads_existing_collection() -> None:
    client = QdrantClient(location=":memory:")

    QdrantVectorStore.create(
        client=client,
        collection_name="test_chunks",
        chunks=chunks(),
        embeddings=np.eye(2, dtype=np.float32),
    )

    restored = QdrantVectorStore.load(
        client=client,
        collection_name="test_chunks",
    )

    results = restored.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert results[0].chunk.id == "guide-0000"
    assert results[0].score == pytest.approx(1.0)


def test_qdrant_store_load_rejects_missing_collection() -> None:
    client = QdrantClient(location=":memory:")

    with pytest.raises(
        ValueError,
        match="collection does not exist",
    ):
        QdrantVectorStore.load(
            client=client,
            collection_name="missing",
        )

def test_qdrant_store_survives_client_restart(tmp_path) -> None:
    storage_path = tmp_path / "qdrant"

    first_client = QdrantClient(
        path=str(storage_path),
    )

    QdrantVectorStore.create(
        client=first_client,
        collection_name="persistent_chunks",
        chunks=chunks(),
        embeddings=np.eye(2, dtype=np.float32),
    )

    first_client.close()

    second_client = QdrantClient(
        path=str(storage_path),
    )

    restored = QdrantVectorStore.load(
        client=second_client,
        collection_name="persistent_chunks",
    )

    results = restored.search(
        np.array([0.0, 1.0], dtype=np.float32),
        top_k=1,
    )

    assert results[0].chunk.id == "guide-0001"
    assert results[0].chunk.text == "Docker"

    second_client.close()
