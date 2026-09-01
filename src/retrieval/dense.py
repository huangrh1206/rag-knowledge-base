from typing import Protocol

import numpy as np

from src.rag.models import SearchResult
from src.persistence.protocol import SearchStore
import logging

logger = logging.getLogger(__name__)

class QueryEmbedder(Protocol):
    # src\embeddings.py
    def embed_query(self, text: str) -> np.ndarray:
        ...

class Retriever:
    def __init__(
        self,
        embedder: QueryEmbedder,
        store: SearchStore,
        threshold: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._threshold = threshold

    def search_candidates(
        self,
        question: str,
        limit: int,
    ) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("question cannot be empty")

        if limit <= 0:
            raise ValueError("limit must be positive")

        query = self._embedder.embed_query(question)

        candidates = self._store.search(
            query,
            limit,
        )

        return [
            result
            for result in candidates
            if result.score >= self._threshold
        ]

    def search(
        self, 
        question: str,
        limit: int,
    ) -> list[SearchResult]:
        """Compatibility alias for Dense-only retrieval."""
        return self.search_candidates(
            question,
            limit=limit,
        )

def format_evidence(
    results: list[SearchResult],
) -> str:
    blocks: list[str] = []

    for number, result in enumerate(results, start=1):
        chunk = result.chunk

        blocks.append(
            f"[{number}] 来源：{chunk.source}，"
            f"第 {chunk.paragraph_start}-{chunk.paragraph_end} 段\n"
            f"内容：{chunk.text}"
        )

    return "\n\n".join(blocks)
