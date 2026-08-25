import pytest

from evaluation.metrics import evaluate_retrieval

def  test_evaluation_calculates_recall_and_mrr() -> None:
    cases = [
        {
           "id": "q001",
            "category": "answerable",
            "expected_sources": ["guide.docx"], 
        },
        {
            "id": "q002",
            "category": "answerable",
            "expected_sources": ["api.docx"],
        },
        {
            "id": "q003",
            "category": "unanswerable",
            "expected_sources": [],
        },
    ]


    summary = evaluate_retrieval(
        cases=cases,
        retrieved_sources={
            "q001": ["guide.docx", "other.docx"],
            "q002": ["other.docx", "api.docx"],
        },
        refusal_predictions={
            "q003": True,
        },
        evidence_predictions={
            "q001": True,
            "q002": True,
        },
    )

    assert summary.answerable_count == 2
    assert summary.recall_at_1 == pytest.approx(0.5)
    assert summary.recall_at_5 == pytest.approx(1.0)
    assert summary.mrr == pytest.approx(0.75)
    assert summary.unanswerable_count == 1
    assert summary.refusal_accuracy == pytest.approx(1.0)
    assert summary.answerable_acceptance == pytest.approx(1.0)

def test_missing_refusal_prediction_counts_as_incorrect() -> None:
    cases = [
        {
            "id": "q001",
            "category": "answerable",
            "expected_sources": ["guide.docx"],
        },
        {
            "id": "q002",
            "category": "unanswerable",
            "expected_sources": [],
        },
    ]

    summary = evaluate_retrieval(
        cases=cases,
        retrieved_sources={"q001": ["guide.docx"]},
        refusal_predictions={},
        evidence_predictions={
            "q001": True,
        },
    )

    assert summary.refusal_accuracy == 0.0
    assert summary.answerable_acceptance == pytest.approx(1.0)

def test_evaluation_requires_both_case_types() -> None:
    with pytest.raises(ValueError, match="answerable"):
        evaluate_retrieval(
            cases = [
                {
                    "id": "q001",
                    "category": "unanswerable",
                    "expected_sources": [],
                }           
            ],
            retrieved_sources={},
            refusal_predictions={},
            evidence_predictions={},
        )

    with pytest.raises(ValueError, match="unanswerable"):
        evaluate_retrieval(
            cases=[
                {
                    "id": "q001",
                    "category": "answerable",
                    "expected_sources": ["guide.docx"],
                }
            ],
            retrieved_sources={},
            refusal_predictions={},
            evidence_predictions={},
        )

