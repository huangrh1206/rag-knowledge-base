from typing import Protocol

import numpy as np

from src.models import SearchResult


class QueryEmbedder(Protocol):
    # src\embeddings.py
    def embed_query(self, text: str) -> np.ndarray:
        ...


class SearchStore(Protocol):
    # src\vector_store.py
    def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        ...


class Retriever:
    def __init__(
        self,
        embedder: QueryEmbedder,
        store: SearchStore,
        top_k: int,
        threshold: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._threshold = threshold

    def search(self, question: str) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("question cannot be empty")
        query = self._embedder.embed_query(question)
        results = self._store.search(
            query,
            self._top_k,
        )

        return [
            result
            for result in results
            if result.score >= self._threshold
        ]


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
