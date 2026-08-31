import pytest

from src.fusion import reciprocal_rank_fusion
from src.models import Chunk, SearchResult


def make_result(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            text=f"text-{chunk_id}",
            source="guide.docx",
            paragraph_start=1,
            paragraph_end=1,
        ),
        score=0.5,
    )


def test_rrf_merges_and_deduplicates_by_chunk_id() -> None:
    dense = [
        make_result("chunk-a"),
        make_result("chunk-b"),
        make_result("chunk-c"),
    ]

    bm25 = [
        make_result("chunk-b"),
        make_result("chunk-d"),
        make_result("chunk-e"),
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense, bm25],
        rank_constant=1,
    )

    ids = [
        item.result.chunk.id
        for item in fused
    ]

    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert set(ids) == {
        "chunk-a",
        "chunk-b",
        "chunk-c",
        "chunk-d",
        "chunk-e",
    }


def test_rrf_rewards_results_present_in_both_lists() -> None:
    dense = [
        make_result("chunk-a"),
        make_result("chunk-b"),
    ]

    bm25 = [
        make_result("chunk-b"),
        make_result("chunk-c"),
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense, bm25],
        rank_constant=1,
    )

    assert fused[0].result.chunk.id == "chunk-b"
    assert fused[0].fusion_score == pytest.approx(
        1 / 3 + 1 / 2
    )


def test_rrf_supports_weights() -> None:
    dense = [
        make_result("chunk-a"),
    ]

    bm25 = [
        make_result("chunk-b"),
    ]

    fused = reciprocal_rank_fusion(
        ranked_lists=[dense, bm25],
        weights=[2.0, 1.0],
        rank_constant=1,
    )

    assert fused[0].result.chunk.id == "chunk-a"
    assert fused[0].fusion_score == pytest.approx(1.0)


def test_rrf_rejects_invalid_weights() -> None:
    with pytest.raises(
        ValueError,
        match="weights must match",
    ):
        reciprocal_rank_fusion(
            ranked_lists=[
                [make_result("chunk-a")],
                [make_result("chunk-b")],
            ],
            weights=[1.0],
        )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        reciprocal_rank_fusion(
            ranked_lists=[
                [make_result("chunk-a")],
            ],
            weights=[-1.0],
        )