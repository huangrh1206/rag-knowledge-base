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

        if case["category"] == "unanswerable":
            refusal_predictions[case_id] = not results

        details.append(
            {
                "id": case_id,
                "category": case["category"],
                "retrieved_sources": sources,
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