from src.rag.evidence_policy import EvidencePolicy
from src.rag.models import Chunk, SearchResult

def result(score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id="chunk-1",
            text="内容",
            source="guide.docx",
            paragraph_start=1,
            paragraph_end=2,
        ),
        score=score,
    )

def test_policy_rejects_empty_results() -> None:
    decision = EvidencePolicy(0.6).evaluate([])

    assert decision.allowed is False
    assert decision.reason == "not_enough_results"


def test_policy_rejects_low_score() -> None:
    decision = EvidencePolicy(0.6).evaluate([result(0.4)])

    assert decision.allowed is False
    assert decision.reason == "top_score_below_minimum"


def test_policy_accepts_sufficient_score() -> None:
    decision = EvidencePolicy(0.6).evaluate([result(0.8)])

    assert decision.allowed is True
    assert decision.reason == "score_sufficient"

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

def test_policy_uses_highest_vector_score_after_reranking() -> None:
    results = [
        make_result("chunk-c", 0.55),
        make_result("chunk-a", 0.80),
        make_result("chunk-b", 0.70),
    ]

    policy = EvidencePolicy(
        minimum_score=0.60,
        minimum_results=1,
    )

    decision = policy.evaluate(results)

    assert decision.allowed is True
    assert decision.reason == "score_sufficient"

def test_policy_rejects_when_all_vector_scores_are_low() -> None:
    results = [
        make_result("chunk-a", 0.55),
        make_result("chunk-b", 0.40),
    ]

    policy = EvidencePolicy(
        minimum_score=0.60,
        minimum_results=1,
    )

    decision = policy.evaluate(results)

    assert decision.allowed is False
    assert decision.reason == "top_score_below_minimum"
