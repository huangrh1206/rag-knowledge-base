from typing import Any

from qdrant_client import QdrantClient

from src.config import Settings
from src.qdrant_vector_store import QdrantVectorStore
from src.store_protocol import SearchStore
from src.vector_store import VectorStore

def load_search_store(
    settings: Settings,
    qdrant_client: QdrantClient | None = None,
) -> SearchStore:
    if settings.vector_store_backend == "numpy":
        return VectorStore.load(settings.index_dir)

    if settings.vector_store_backend == "qdrant":
        client = qdrant_client or QdrantClient(
            path=str(settings.qdrant_path),
        )
        return QdrantVectorStore.load(
            client=client,
            collection_name=settings.qdrant_collection,
        )

    raise ValueError(
         f"unsupported vector store backend: "
         f"{settings.vector_store_backend}"
    )
