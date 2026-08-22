import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.api_client import create_openai_client
from src.config import Settings
from src.embeddings import EmbeddingClient
from src.retriever import Retriever
from src.vector_store import VectorStore

from evaluation.metrics import evaluate_retrieval

def load_questions(path: Path) -> list[dict[str, object]]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, list):
        raise ValueError("evaluation file must contain a list")

    return value

def build_retriever(settings: Settings) -> Retriever:
    client = create_openai_client(settings)
    embedder = EmbeddingClient(
        api=client.embeddings,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    store = VectorStore.load(settings.index_dir)

    return Retriever(
        embedder=embedder,
        store=store,
        top_k=settings.top_k,
        threshold=settings.similarity_threshold,
    )

def run_evaluation(
    questions: list[dict[str, object]],
    retriever: Retriever,
) -> dict[str, object]:
    retrieved_sources: dict[str, list[str]] = {}
    refusal_predictions: dict[str, bool] = {}
    details: list[dict[str, object]] = []

    for case in questions:
        case_id = case["id"]
        question = case["question"]
        results = retriever.search(question)
        sources = [
            result.chunk.source
            for result in results
        ]

        retrieved_sources[case_id] = sources

        top_score = results[0].score if results else 0.0
        second_score = results[1].score if len(results) > 1 else 0.0

        refused = False
        refusal_reason = None

        if case["category"] == "unanswerable":
            refused = (
                top_score < 0.60
                or top_score - second_score < 0.05
            )
            if top_score < 0.60:
                refusal_reason = "top_score_below_threshold"
            elif top_score - second_score < 0.05:
                refusal_reason = "score_margin_too_small"
            else:
                refusal_reason = "evidence_looks_sufficient"
            refusal_predictions[case_id] = refused

        details.append(
            {
                "id": case_id,
                "category": case["category"],
                "retrieved_sources": sources,
                "top_score": round(top_score, 4),
                "second_score": round(second_score, 4),
                "refused": refused,
                "refusal_reason": refusal_reason,
                "scores": [
                    round(result.score, 4)
                    for result in results
                ],
            }
        )

    summary = evaluate_retrieval(
        cases=questions,
        retrieved_sources=retrieved_sources,
        refusal_predictions=refusal_predictions,
    )

    return {
        "summary": asdict(summary),
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
    report = run_evaluation(
        questions=questions,
        retriever=retriever,
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