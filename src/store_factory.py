from typing import Any

from qdrant_client import QdrantClient

from src.config import Settings
from src.qdrant_vector_store import QdrantVectorStore
from src.store_protocol import SearchStore
from src.vector_store import VectorStore
from src.index_manifest import (
    ManifestError,
    load_manifest,
    qdrant_manifest_directory,
    validate_compatibility,
    validate_vector_dimension,
)

def load_search_store(
    settings: Settings,
    qdrant_client: QdrantClient | None = None,
) -> SearchStore:
    if settings.vector_store_backend == "numpy":
        manifest = load_manifest(settings.index_dir)

        validate_compatibility(
            manifest,
            backend=settings.vector_store_backend,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        store = VectorStore.load(settings.index_dir)

        validate_vector_dimension(
            manifest,
            store.vector_dimension,
        )

        return store

    if settings.vector_store_backend == "qdrant":
        manifest_dir = qdrant_manifest_directory(
            qdrant_path=settings.qdrant_path,
            collection_name=settings.qdrant_collection,
        )
        manifest = load_manifest(manifest_dir)

        validate_compatibility(
            manifest,
            backend="qdrant",
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        client = qdrant_client or QdrantClient(
            path=str(settings.qdrant_path),
        )
        store = QdrantVectorStore.load(
            client=client,
            collection_name=settings.qdrant_collection,
        )

        validate_vector_dimension(
            manifest,
            store.vector_dimension,
        )

        return store

    raise ValueError(
         f"unsupported vector store backend: "
         f"{settings.vector_store_backend}"
    )
