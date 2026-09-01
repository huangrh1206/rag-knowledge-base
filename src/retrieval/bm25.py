from typing import Protocol
from rank_bm25 import BM25Okapi

from src.rag.models import SearchResult


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]:
        ...


class CharacterTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return [
            character
            for character in text.lower()
            if not character.isspace()
        ]


class BM25Retriever:
    def __init__(
        self,
        results: list[SearchResult],
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._results = list(results)
        self._tokenizer = tokenizer or CharacterTokenizer()

        corpus = [
            self._tokenizer.tokenize(
                result.chunk.text
            )
            for result in self._results
        ]

        self._bm25 = BM25Okapi(corpus)

    def search(
        self,
        question: str,
        top_k: int,
    ) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = self._tokenizer.tokenize(question)
        scores = self._bm25.get_scores(query_tokens)

        indexes = sorted(
            range(len(self._results)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[: min(top_k, len(self._results))]

        return [
            self._results[index]
            for index in indexes
        ]
