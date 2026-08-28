from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class EvaluationSummary:
    answerable_count: int
    recall_at_1: float
    recall_at_5: float
    mrr: float
    answerable_acceptance: float
    keyword_coverage: float
    unanswerable_count: int
    refusal_accuracy: float

def _hit_at_k(
    retrieved_sources: list[str],
    expected_sources: set[str],
    k: int
) -> bool:
    return bool(
        expected_sources.intersection(retrieved_sources[:k])
    )

def _reciprocal_rank(
    retrieved_sources: list[str],
    expected_sources: list[str],
) -> float:
    for rank, source in enumerate(retrieved_sources, start = 1):
        if source in expected_sources:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval(
    cases: Iterable[dict[str, object]],
    retrieved_sources: dict[str, list[str]],
    refusal_predictions: dict[str, bool],
    evidence_predictions: dict[str, bool],
    keyword_coverages: dict[str, float] | None = None,
) -> EvaluationSummary:
    answerable = [
        case
        for case in cases
        if case["category"] == "answerable"
    ]

    accepted_answerable = 0

    for case in answerable:
        case_id = str(case["id"])

        accepted_answerable += int(
            evidence_predictions.get(case_id, False)
        )

    unanswerable = [
        case
        for case in cases
        if case["category"] == "unanswerable"
    ]

    if not answerable:
        raise ValueError("evaluation requires answerable cases")

    if not unanswerable:
        raise ValueError("evaluation requires unanswerable cases")

    keyword_coverages = keyword_coverages or {}
    average_keyword_coverage = sum(
        keyword_coverages.get(str(case["id"]), 0.0)
        for case in answerable
    ) / len(answerable)

    recall_1_hits = 0
    recall_5_hits = 0
    reciprocal_ranks: list[float] = []

    for case in answerable:
        case_id = str(case["id"])
        expected_sources = {
            str(source)
            for source in case["expected_sources"]
        }
        sources = retrieved_sources.get(case_id, [])

        recall_1_hits += int(
            _hit_at_k(sources, expected_sources, 1)
        )
        recall_5_hits += int(
            _hit_at_k(sources, expected_sources, 5)
        )
        reciprocal_ranks.append(
            _reciprocal_rank(sources, expected_sources)
        )

    corrent_refusals = 0

    for case in unanswerable:
        case_id = str(case["id"])
        corrent_refusals += int(
            refusal_predictions.get(case_id, False)
        )

    return EvaluationSummary(
        answerable_count=len(answerable),
        recall_at_1=recall_1_hits / len(answerable),
        recall_at_5=recall_5_hits / len(answerable),
        mrr=sum(reciprocal_ranks) / len(answerable),
        answerable_acceptance=accepted_answerable / len(answerable),
        keyword_coverage=average_keyword_coverage,
        unanswerable_count=len(unanswerable),
        refusal_accuracy=corrent_refusals / len(unanswerable),
    )