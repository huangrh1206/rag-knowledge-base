import json

import numpy as np
import pytest

from src.rag.models import Chunk
from src.persistence.numpy_store import IndexFormatError, VectorStore


def sample_chunks() -> list[Chunk]:
    return [
        Chunk("guide-0000", "FastAPI", "guide.docx", 1, 1),
        Chunk("guide-0001", "Docker", "guide.docx", 2, 2),
    ]


def test_cosine_search_returns_highest_score_first() -> None:
    store = VectorStore(
        sample_chunks(),
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )

    results = store.search(
        np.array([0.9, 0.1], dtype=np.float32),
        top_k=2,
    )

    assert [result.chunk.id for result in results] == [
        "guide-0000",
        "guide-0001",
    ]
    assert results[0].score > results[1].score


def test_vector_store_rejects_chunk_embedding_count_mismatch() -> None:
    with pytest.raises(ValueError, match="same number"):
        VectorStore(
            sample_chunks(),
            np.array([[1.0, 0.0]], dtype=np.float32),
        )


def test_search_rejects_non_positive_top_k() -> None:
    store = VectorStore(
        sample_chunks(),
        np.eye(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        store.search(
            np.array([1.0, 0.0], dtype=np.float32),
            top_k=0,
        )


def test_search_rejects_query_dimension_mismatch() -> None:
    store = VectorStore(
        sample_chunks(),
        np.eye(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="query dimension must match"):
        store.search(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            top_k=1,
        )


def test_vector_store_requires_two_dimensional_embeddings() -> None:
    with pytest.raises(ValueError, match="embeddings must be a 2-D matrix"):
        VectorStore(
            sample_chunks(),
            np.array([1.0, 0.0], dtype=np.float32),
        )


def test_save_writes_chunk_metadata_and_embeddings(tmp_path) -> None:
    store = VectorStore(
        sample_chunks(),
        np.eye(2, dtype=np.float32),
    )

    store.save(tmp_path)

    metadata = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    embeddings = np.load(tmp_path / "embeddings.npy")

    assert metadata[0]["id"] == "guide-0000"
    assert metadata[1]["source"] == "guide.docx"
    np.testing.assert_array_equal(embeddings, np.eye(2, dtype=np.float32))


def test_load_restores_saved_vector_store(tmp_path) -> None:
    chunks = sample_chunks()
    embeddings = np.eye(2, dtype=np.float32)
    VectorStore(chunks, embeddings).save(tmp_path)

    restored = VectorStore.load(tmp_path)

    assert restored.chunks == chunks
    np.testing.assert_array_equal(restored.embeddings, embeddings)


def test_load_reports_invalid_index_for_malformed_metadata(tmp_path) -> None:
    (tmp_path / "chunks.json").write_text(
        "not valid json",
        encoding="utf-8",
    )
    np.save(tmp_path / "embeddings.npy", np.eye(2, dtype=np.float32))

    with pytest.raises(IndexFormatError, match="invalid vector index"):
        VectorStore.load(tmp_path)


def test_load_reports_invalid_index_when_embeddings_are_missing(tmp_path) -> None:
    metadata = [chunk.to_dict() for chunk in sample_chunks()]
    (tmp_path / "chunks.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(IndexFormatError, match="invalid vector index"):
        VectorStore.load(tmp_path)


def test_load_reports_invalid_index_for_mismatched_item_counts(tmp_path) -> None:
    metadata = [chunk.to_dict() for chunk in sample_chunks()]
    (tmp_path / "chunks.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    np.save(
        tmp_path / "embeddings.npy",
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(IndexFormatError, match="invalid vector index"):
        VectorStore.load(tmp_path)


def test_load_reports_invalid_index_for_incomplete_chunk_metadata(tmp_path) -> None:
    (tmp_path / "chunks.json").write_text(
        json.dumps([{"id": "broken"}]),
        encoding="utf-8",
    )
    np.save(
        tmp_path / "embeddings.npy",
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(IndexFormatError, match="invalid vector index"):
        VectorStore.load(tmp_path)


def test_save_preserves_existing_index_when_vector_write_fails(
    tmp_path,
    monkeypatch,
) -> None:
    original = VectorStore(
        sample_chunks(),
        np.eye(2, dtype=np.float32),
    )
    original.save(tmp_path)

    replacement_chunks = [
        Chunk("new-0000", "new text", "new.docx", 1, 1),
    ]
    replacement = VectorStore(
        replacement_chunks,
        np.array([[0.5, 0.5]], dtype=np.float32),
    )

    def fail_vector_write(*args, **kwargs) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(np, "save", fail_vector_write)

    with pytest.raises(OSError, match="simulated disk failure"):
        replacement.save(tmp_path)

    metadata = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in metadata] == [
        "guide-0000",
        "guide-0001",
    ]
    np.testing.assert_array_equal(
        np.load(tmp_path / "embeddings.npy"),
        np.eye(2, dtype=np.float32),
    )

def test_vector_store_exposes_vector_dimension() -> None:
    store = VectorStore(
        sample_chunks(),
        np.eye(2, dtype=np.float32),
    )

    assert store.vector_dimension == 2
