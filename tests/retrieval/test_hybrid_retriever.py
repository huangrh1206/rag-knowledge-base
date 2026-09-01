import logging

import pytest

from src.retrieval.hybrid import HybridRetriever
from src.rag.models import Chunk, SearchResult
from src.retrieval.reranker import DisabledReranker


class FakeDenseRetriever:
    def __init__(self, results):
        self.results = results
        self.requested_limit = None

    def search_candidates(self, question, limit):
        self.requested_limit = limit
        return list(self.results)


class FakeBM25Retriever:
    def __init__(self, results):
        self.results = results
        self.requested_top_k = None

    def search(self, question, top_k):
        self.requested_top_k = top_k
        return list(self.results)


class ReverseReranker:
    def rerank(self, question, results):
        return list(reversed(results))


class FailingReranker:
    def rerank(self, question, results):
        raise RuntimeError("reranker unavailable")


def make_result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            text=f"text-{chunk_id}",
            source="guide.docx",
            paragraph_start=1,
            paragraph_end=2,
        ),
        score=score,
    )


def build_retriever(
    dense_results,
    bm25_results,
    *,
    candidate_k=3,
    top_k=2,
    reranker=None,
    dense_weight=1.0,
    bm25_weight=1.0,
):
    dense = FakeDenseRetriever(dense_results)
    bm25 = FakeBM25Retriever(bm25_results)

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        candidate_k=candidate_k,
        top_k=top_k,
        reranker=reranker or DisabledReranker(),
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )

    return retriever, dense, bm25


def test_hybrid_retriever_calls_both_retrievers_with_candidate_k():
    dense_results = [
        make_result("dense-a", 0.95),
        make_result("dense-b", 0.90),
    ]
    bm25_results = [
        make_result("bm25-a", 3.0),
        make_result("bm25-b", 2.0),
    ]

    retriever, dense, bm25 = build_retriever(
        dense_results,
        bm25_results,
        candidate_k=5,
        top_k=2,
    )

    retriever.search("test question")

    assert dense.requested_limit == 5
    assert bm25.requested_top_k == 5
    assert retriever.last_dense_chunk_ids == ["dense-a", "dense-b"]
    assert retriever.last_bm25_chunk_ids == ["bm25-a", "bm25-b"]


def test_hybrid_retriever_returns_final_top_k():
    dense_results = [
        make_result("chunk-a", 0.95),
        make_result("chunk-b", 0.90),
        make_result("chunk-c", 0.85),
    ]

    bm25_results = [
        make_result("chunk-c", 3.0),
        make_result("chunk-b", 2.0),
        make_result("chunk-a", 1.0),
    ]

    retriever, _, _ = build_retriever(
        dense_results,
        bm25_results,
        candidate_k=3,
        top_k=2,
    )

    results = retriever.search("test question")

    assert len(results) == 2


def test_hybrid_retriever_supports_dense_only_mode():
    dense_results = [
        make_result("dense-a", 0.95),
        make_result("dense-b", 0.90),
    ]

    bm25_results = [
        make_result("bm25-a", 3.0),
    ]

    retriever, dense, bm25 = build_retriever(
        dense_results,
        bm25_results,
        candidate_k=3,
        top_k=2,
        dense_weight=1.0,
        bm25_weight=0.0,
    )

    results = retriever.search("test question")

    assert dense.requested_limit == 3
    assert bm25.requested_top_k is None
    assert len(results) == 2


def test_hybrid_retriever_supports_bm25_only_mode():
    dense_results = [
        make_result("dense-a", 0.95),
    ]

    bm25_results = [
        make_result("bm25-a", 3.0),
        make_result("bm25-b", 2.0),
    ]

    retriever, dense, bm25 = build_retriever(
        dense_results,
        bm25_results,
        candidate_k=3,
        top_k=2,
        dense_weight=0.0,
        bm25_weight=1.0,
    )

    results = retriever.search("test question")

    assert dense.requested_limit is None
    assert bm25.requested_top_k == 3
    assert [item.chunk.id for item in results] == [
        "bm25-a",
        "bm25-b",
    ]


def test_hybrid_retriever_applies_reranker_after_fusion():
    dense_results = [
        make_result("chunk-a", 0.95),
        make_result("chunk-b", 0.90),
        make_result("chunk-c", 0.85),
    ]

    bm25_results = [
        make_result("chunk-a", 3.0),
        make_result("chunk-b", 2.0),
        make_result("chunk-c", 1.0),
    ]

    retriever, _, _ = build_retriever(
        dense_results,
        bm25_results,
        candidate_k=3,
        top_k=2,
        reranker=ReverseReranker(),
    )

    results = retriever.search("test question")

    assert [item.chunk.id for item in results] == [
        "chunk-c",
        "chunk-b",
    ]


def test_hybrid_retriever_keeps_citation_metadata_after_rerank():
    dense_results = [
        make_result("chunk-a", 0.95),
        make_result("chunk-b", 0.90),
    ]

    bm25_results = [
        make_result("chunk-b", 3.0),
        make_result("chunk-a", 2.0),
    ]

    retriever, _, _ = build_retriever(
        dense_results,
        bm25_results,
        candidate_k=2,
        top_k=2,
        reranker=ReverseReranker(),
    )

    results = retriever.search("test question")

    assert results[0].chunk.id == "chunk-b"
    assert results[0].chunk.source == "guide.docx"
    assert results[0].chunk.paragraph_start == 1
    assert results[0].chunk.paragraph_end == 2


def test_hybrid_retriever_falls_back_when_reranker_fails(caplog):
    dense_results = [
        make_result("chunk-a", 0.95),
        make_result("chunk-b", 0.90),
    ]

    bm25_results = [
        make_result("chunk-a", 3.0),
        make_result("chunk-b", 2.0),
    ]

    retriever, _, _ = build_retriever(
        dense_results,
        bm25_results,
        candidate_k=2,
        top_k=2,
        reranker=FailingReranker(),
    )

    with caplog.at_level(
        logging.ERROR,
        logger="src.retrieval.hybrid",
    ):
        results = retriever.search("test question")

    assert len(results) == 2
    assert any(
        "rerank failed; fallback to vector results"
        in record.getMessage()
        for record in caplog.records
    )


def test_hybrid_retriever_rejects_invalid_parameters():
    dense = FakeDenseRetriever([])
    bm25 = FakeBM25Retriever([])

    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than or equal to top_k",
    ):
        HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            candidate_k=2,
            top_k=3,
        )

    with pytest.raises(ValueError, match="at least one fusion weight"):
        HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            candidate_k=3,
            top_k=2,
            dense_weight=0.0,
            bm25_weight=0.0,
        )
