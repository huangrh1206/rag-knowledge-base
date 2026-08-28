from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from src.models import Chunk, SearchResult

class QdrantVectorStore:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        if vector_size <= 0:
            raise ValueError("vector_size must be positive")

        self._client = client
        self._collection_name = collection_name
        self._vector_size = vector_size

    @property
    def vector_dimension(self) -> int:
        return self._vector_size


    @classmethod
    def create(
        cls,
        client: QdrantClient,
        collection_name: str,
        chunks: list[Chunk],
        embeddings,
    ) -> "QdrantVectorStore":
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D matrix")

        if not chunks:
            raise ValueError("at least one chunk is required")

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "chunks and embeddings must have the same number of items"
            )

        vector_size = embeddings.shape[1]

        if vector_size <= 0:
            raise ValueError("embedding dimension must be positive")

        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        )

        points = [
            models.PointStruct(
                id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{collection_name}:{chunk.id}"
                    )
                ),
                vector=embedding.tolist(),
                payload=chunk.to_dict()
            )
            for chunk, embedding in zip(
                chunks,
                embeddings,
                strict=True,
            )
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        return cls(
            client=client,
            collection_name=collection_name,
            vector_size=vector_size,
        )

    @classmethod
    def load(
        cls,
        client: QdrantClient,
        collection_name: str,
    ) -> "QdrantVectorStore":
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        if not client.collection_exists(collection_name):
            raise ValueError(
                f"Qdrant collection does not exist: {collection_name}"
            )

        collection = client.get_collection(collection_name)
        vector_size = collection.config.params.vectors.size

        if not isinstance(vector_size, int) or vector_size <= 0:
            raise ValueError(
                "Qdrant collection has invalid vector size"
            )

        return cls(
            client=client,
            collection_name=collection_name,
            vector_size=vector_size,
        )

    def search(
        self, 
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if query.ndim != 1 or query.shape[0] != self._vector_size:
            raise ValueError("query dimension must match embedding dimension")

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query.tolist(),
            limit=top_k,
            with_payload=True,
        )
        
        results: list[SearchResult] = []

        for point in response.points:
            if point.payload is None:
                raise ValueError(
                    "Qdrant result is missing chunk payload"
                )
            results.append(
                SearchResult(
                    chunk=Chunk.from_dict(point.payload),
                    score=float(point.score),
                )
            )
        return results
