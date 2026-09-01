from types import SimpleNamespace

import numpy as np
from qdrant_client import QdrantClient

from src.config import Settings
from src.retrieval.hybrid import HybridRetriever
from src.persistence.manifest import (
    IndexManifest,
    qdrant_manifest_directory,
    write_manifest,
)
from src.rag.models import Chunk
from src.persistence.qdrant_store import QdrantVectorStore
from src.retrieval.factory import create_retriever


def test_factory_builds_hybrid_retriever_for_qdrant(tmp_path) -> None:
    qdrant_path = tmp_path / "qdrant"
    qdrant_client = QdrantClient(path=str(qdrant_path))

    QdrantVectorStore.create(
        client=qdrant_client,
        collection_name="rag_chunks",
        chunks=[
            Chunk("chunk-a", "FastAPI routing", "guide.docx", 1, 1),
            Chunk("chunk-b", "Docker deployment", "guide.docx", 2, 2),
            Chunk("chunk-c", "Python typing", "guide.docx", 3, 3),
        ],
        embeddings=np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
            dtype=np.float32,
        ),
    )
    qdrant_client.close()

    write_manifest(
        qdrant_manifest_directory(qdrant_path, "rag_chunks"),
        IndexManifest(
            schema_version=1,
            backend="qdrant",
            embedding_model="embedding-model",
            vector_dimension=2,
            chunk_size=700,
            chunk_overlap=100,
            document_count=1,
            chunk_count=3,
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
        hybrid_enabled=True,
        rerank_enabled=False,
        retrieval_candidate_k=3,
        top_k=1,
    )

    retriever = create_retriever(
        client=SimpleNamespace(embeddings=object()),
        settings=settings,
    )

    assert isinstance(retriever, HybridRetriever)
    bm25_results = retriever._bm25_retriever.search(
        "Docker",
        top_k=1,
    )
    assert bm25_results[0].chunk.id == "chunk-b"
