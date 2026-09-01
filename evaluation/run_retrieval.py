import argparse
import json
from dataclasses import asdict
from pathlib import Path
import time
from typing import Protocol

from openai import OpenAI

from evaluation.metrics import evaluate_retrieval
from src.models import SearchResult
from src.api_client import create_openai_client
from src.config import Settings
from src.evidence_policy import EvidencePolicy
from src.retriever_factory import create_retriever

class EvaluationRetriever(Protocol):
    def search(
        self,
        question: str,
    ) -> list[SearchResult]:
        ...

def load_questions(path: Path) -> list[dict[str, object]]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, list):
        raise ValueError("evaluation file must contain a list")

    return value

def build_retriever(settings: Settings) -> EvaluationRetriever:
    client = create_openai_client(settings)

    return create_retriever(
        client=client,
        settings=settings,
    )

def calculate_keyword_coverage(
    keywords: object,
    results: list[SearchResult],
) -> float:
    if not isinstance(keywords, list) or not keywords:
        return 1.0

    evidence_text = "\n".join(
        result.chunk.text.lower()
        for result in results
    )

    matched = sum(
        1
        for keyword in keywords
        if str(keyword).lower() in evidence_text
    )

    return matched / len(keywords)

def run_evaluation(
    questions: list[dict[str, object]],
    retriever: EvaluationRetriever,
    evidence_policy: EvidencePolicy,
) -> dict[str, object]:
    retrieved_sources: dict[str, list[str]] = {}
    refusal_predictions: dict[str, bool] = {}
    evidence_predictions: dict[str, bool] = {}
    details: list[dict[str, object]] = []
    keyword_coverages: dict[str, float] = {}
    retrieval_times_ms: list[float] = []

    for case in questions:
        case_id = str(case["id"])
        question = str(case["question"])

        started = time.perf_counter()

        results = retriever.search(question)
        retrieval_ms = (
            time.perf_counter() - started
        ) * 1000
        retrieval_times_ms.append(retrieval_ms)

        keywords_coverage = calculate_keyword_coverage(
            keywords=case.get("expected_keywords", []),
            results=results,
        )
        keyword_coverages[case_id] = keywords_coverage
        
        sources = [
            result.chunk.source
            for result in results
        ]

        chunk_ids = [
            result.chunk.id
            for result in results
        ]

        dense_trace = getattr(retriever, "last_dense_chunk_ids", None)
        dense_chunk_ids = list(
            chunk_ids if dense_trace is None else dense_trace
        )
        bm25_chunk_ids = list(
            getattr(retriever, "last_bm25_chunk_ids", [])
        )

        paragraph_spans = [
            {
                "source": result.chunk.source,
                "start": result.chunk.paragraph_start,
                "end": result.chunk.paragraph_end,
            }
            for result in results
        ]

        retrieved_sources[case_id] = sources

        first_result_vector_score = (
            results[0].score
            if results
            else 0.0
        )

        max_vector_score = max(
            (
                result.score
                for result in results
            ),
            default=0.0,
        )

        rerank_scores = [
            (
                round(result.rerank_score, 4)
                if result.rerank_score is not None
                else None
            )
            for result in results
        ]

        decision = evidence_policy.evaluate(results)
        accepted = decision.allowed
        refused = not accepted

        if case["category"] == "answerable":
            evidence_predictions[case_id] = accepted

        if case["category"] == "unanswerable":
            refusal_predictions[case_id] = refused

        details.append(
            {
                "id": case_id,
                "category": case["category"],
                "retrieved_sources": sources,
                "retrieved_chunk_ids": chunk_ids,
                "dense_chunk_ids": dense_chunk_ids,
                "bm25_chunk_ids": bm25_chunk_ids,
                "retrieved_paragraph_spans": paragraph_spans,
                "first_result_vector_score": round(
                    first_result_vector_score,
                    4,
                ),
                "evidence_allowed": accepted,
                "keyword_coverage": round(keywords_coverage, 4),
                "refused": refused,
                "refusal_reason": decision.reason,
                "scores": [
                    round(result.score, 4)
                    for result in results
                ],
                "rerank_scores": rerank_scores,
                "retrieval_ms": round(retrieval_ms, 2),
            }
        )

    summary = evaluate_retrieval(
        cases=questions,
        retrieved_sources=retrieved_sources,
        refusal_predictions=refusal_predictions,
        evidence_predictions=evidence_predictions,
        keyword_coverages=keyword_coverages,
    )

    summary_data = asdict(summary)
    summary_data["average_retrieval_ms"] = (
        sum(retrieval_times_ms)
        / len(retrieval_times_ms)
        if retrieval_times_ms
        else 0.0
    )

    return {
        "summary": summary_data,
        "details": details,
    }

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/questions.json")
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    questions = load_questions(args.questions)
    retriever = build_retriever(settings)
    evidence_policy = EvidencePolicy(
        minimum_score=settings.evidence_minimum_score,
        minimum_results=settings.evidence_minimum_results,
    )
    report = run_evaluation(
        questions=questions,
        retriever=retriever,
        evidence_policy = evidence_policy,
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

#python -m evaluation.run_retrieval > evaluation\paraphrase-report-v2.json
