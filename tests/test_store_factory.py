import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.config import Settings
from src.models import Chunk
from src.qdrant_vector_store import QdrantVectorStore
from src.store_factory import load_search_store
from src.vector_store import VectorStore
from src.index_manifest import IndexManifest, write_manifest
from src.index_manifest import (
    IndexManifest,
    qdrant_manifest_directory,
    write_manifest,
)

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

    write_manifest(
        index_dir,
        IndexManifest(
            schema_version=1,
            backend="numpy",
            embedding_model="embedding-model",
            vector_dimension=2,
            chunk_size=700,
            chunk_overlap=100,
            document_count=1,
            chunk_count=1,
        ),
    )    

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

    write_manifest(
        qdrant_manifest_directory(
            qdrant_path=qdrant_path,
            collection_name="rag_chunks",
        ),
        IndexManifest(
            schema_version=1,
            backend="qdrant",
            embedding_model="embedding-model",
            vector_dimension=2,
            chunk_size=700,
            chunk_overlap=100,
            document_count=1,
            chunk_count=1,
        ),
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

def test_factory_rejects_numpy_manifest_model_mismatch(
    tmp_path,
) -> None:
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

    write_manifest(
        index_dir,
        IndexManifest(
            schema_version=1,
            backend="numpy",
            embedding_model="old-model",
            vector_dimension=2,
            chunk_size=700,
            chunk_overlap=100,
            document_count=1,
            chunk_count=1,
        ),
    )

    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat-model",
        embedding_model="new-model",
        vector_store_backend="numpy",
        index_dir=index_dir,
    )

    with pytest.raises(
        ValueError,
        match="embedding model does not match",
    ):
        load_search_store(settings)