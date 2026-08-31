import pytest

from src.models import Chunk, SearchResult
from src.reranker import DisabledReranker
from src.reranker import (
    CrossEncoderReranker,
    DisabledReranker,
)

def make_result(
    chunk_id: str,
    score: float,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            text=f"text-{chunk_id}",
            source="guide.docx",
            paragraph_start=1,
            paragraph_end=1,
        ),
        score=score,
    )


def test_disabled_reranker_keeps_result_order() -> None:
    results = [
        make_result("chunk-a", 0.9),
        make_result("chunk-b", 0.8),
        make_result("chunk-c", 0.7),
    ]

    reranker = DisabledReranker()

    reranked = reranker.rerank(
        question="test question",
        results=results,
    )

    assert [result.chunk.id for result in reranked] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]


def test_disabled_reranker_returns_new_list() -> None:
    results = [
        make_result("chunk-a", 0.9),
    ]

    reranker = DisabledReranker()
    reranked = reranker.rerank("test question", results)

    assert reranked == results
    assert reranked is not results


def test_disabled_reranker_handles_empty_results() -> None:
    reranker = DisabledReranker()

    assert reranker.rerank("test question", []) == []

class FakeCrossEncoder:
    def predict(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        assert pairs == [
            (
                "question",
                "文档标题：guide\n正文：text-chunk-a",
            ),
            (
                "question",
                "文档标题：guide\n正文：text-chunk-b",
            ),
            (
                "question",
                "文档标题：guide\n正文：text-chunk-c",
            ),
        ]

        return [0.10, 0.95, 0.40]

def test_cross_encoder_reranker_orders_and_keeps_both_scores() -> None:
    results = [
        make_result("chunk-a", 0.90),
        make_result("chunk-b", 0.80),
        make_result("chunk-c", 0.70),
    ]

    reranker = CrossEncoderReranker(
        model_name="fake-model",
        model=FakeCrossEncoder(),
    )

    reranked = reranker.rerank(
        question="question",
        results=results,
    )

    assert [
        result.chunk.id
        for result in reranked
    ] == [
        "chunk-b",
        "chunk-c",
        "chunk-a",
    ]

    assert [
        result.score
        for result in reranked
    ] == [
        0.80,
        0.70,
        0.90,
    ]

    assert [
        result.rerank_score
        for result in reranked
    ] == [
        0.95,
        0.40,
        0.10,
    ]

class InvalidCrossEncoder:
    def predict(self, pairs):
        return [0.5]


def test_cross_encoder_rejects_incomplete_scores() -> None:
    reranker = CrossEncoderReranker(
        model_name="fake-model",
        model=InvalidCrossEncoder(),
    )

    with pytest.raises(
        ValueError,
        match="unexpected score count",
    ):
        reranker.rerank(
            "question",
            [
                make_result("chunk-a", 0.9),
                make_result("chunk-b", 0.8),
            ],
        )