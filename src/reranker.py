from typing import Protocol, Sequence
from pathlib import Path
from src.models import SearchResult

class Reranker(Protocol):
    def rerank(
        self,
        question: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Reorder retrieval results for one question."""
        ...

class DisabledReranker:
    """Keep vector retrieval order when reranking is disabled."""

    def rerank(
        self,
        question: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        return list(results)

class CrossEncoderModel(Protocol):
    def predict(
        self,
        pairs: list[tuple[str, str]],
    ) -> Sequence[float]:
        ...

class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        model: CrossEncoderModel | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError(
                "reranker model cannot be empty"
            )

        if model is not None:
            self._model = model
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required "
                "when reranking is enabled"
            ) from exc

        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [
            (
                question, 
                (
                    f"文档标题：{Path(result.chunk.source).stem}\n"
                    f"正文：{result.chunk.text}"
                ),
            )
            for result in results
        ]

        scores = self._model.predict(pairs)

        if len(scores) != len(results):
            raise ValueError(
                "reranker returned an unexpected score count"
            )

        ranked_indexes = sorted(
            range(len(results)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )

        return [
            SearchResult(
                chunk=results[index].chunk,
                score=results[index].score,
                rerank_score=float(scores[index]),
            )
            for index in ranked_indexes
        ]

