import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.models import Chunk
from src.qdrant_vector_store import QdrantVectorStore
from src.vector_store import VectorStore

def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="chunk-1",
            text="FastAPI parameters",
            source="guide.docx",
            paragraph_start=1,
            paragraph_end=2,
        ),
        Chunk(
            id="chunk-2",
            text="Docker deployment",
            source="guide.docx",
            paragraph_start=3,
            paragraph_end=4,
        ),
        Chunk(
            id="chunk-3",
            text="Qdrant vector database",
            source="storage.docx",
            paragraph_start=1,
            paragraph_end=2,
        ),
    ]


def test_numpy_and_qdrant_have_same_search_contract(
    tmp_path,
) -> None:
    chunks = sample_chunks()
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array(
        [0.8, 0.2, 0.0],
        dtype=np.float32,
    )

    numpy_store = VectorStore(
        chunks,
        embeddings,
    )

    qdrant_client = QdrantClient(location=":memory:")

    try:
        qdrant_store = QdrantVectorStore.create(
            client=qdrant_client,
            collection_name="parity_test",
            chunks=chunks,
            embeddings=embeddings,
        )

        numpy_results = numpy_store.search(
            query,
            top_k=3,
        )
        qdrant_results = qdrant_store.search(
            query,
            top_k=3,
        )

        assert [
            result.chunk.id
            for result in numpy_results
        ] == [
            result.chunk.id
            for result in qdrant_results
        ]

        for numpy_result, qdrant_result in zip(
            numpy_results,
            qdrant_results,
            strict=True,
        ):
            assert qdrant_result.score == pytest.approx(
                numpy_result.score,
                abs=1e-5,
            )
    finally:
        qdrant_client.close()