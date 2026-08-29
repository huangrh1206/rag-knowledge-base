from src.evidence_policy import EvidencePolicy
from src.models import Chunk, SearchResult

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