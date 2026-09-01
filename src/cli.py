from datetime import datetime

import argparse
import logging
from pathlib import Path

from src.rag.models import Answer

from openai import OpenAI

from src.infrastructure.openai_client import create_openai_client
from src.config import Settings
from src.infrastructure.embeddings import EmbeddingClient
from src.agent import KnowledgeAgent
from src.rag.generator import AnswerGenerator
from src.rag.service import RAGService, build_index, build_qdrant_index
from src.rag.evidence_policy import EvidencePolicy
from src.retrieval.factory import create_retriever

def configure_logging() -> None:
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / (
        f"rag-{datetime.now().strftime('%Y-%m-%d')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(
                log_file,
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Word document RAG knowledge base"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = subparsers.add_parser(
        "index",
        help="build an index from .docx files",
    )
    index_parser.add_argument(
        "directory",
        type=Path,
    )

    # src\rag_service.py
    ask_parser = subparsers.add_parser(
        "ask",
        help="ask one grounded question"
    )
    ask_parser.add_argument("question")

    subparsers.add_parser(
        "chat",
        help="start an interactive agent chat"
    )
    return parser

def print_answer(answer: Answer) -> None:
    print(answer.answer)

    if answer.citations:
        print("\nSources:")

    for citation in answer.citations:
        print(
            f"[{citation.number}] {citation.source}, "
            f"paragraphs "
            f"{citation.paragraph_start}-"
            f"{citation.paragraph_end}"
        )

def _client(settings: Settings) -> OpenAI:
    return create_openai_client(settings)

def _embedder(
    client: OpenAI,
    settings: Settings,
) -> EmbeddingClient:
    return EmbeddingClient(
        api=client.embeddings,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )


def main() -> int:
    configure_logging()

    args = build_parser().parse_args()

    try:
        settings = Settings.from_env()
        client = _client(settings)

        if args.command == "index":
            embedder = _embedder(client, settings)
            if settings.vector_store_backend == "numpy":
                report = build_index(
                    document_dir=args.directory,
                    index_dir=settings.index_dir,
                    embedder=embedder,
                    embedding_model=settings.embedding_model,
                    chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap,
                )

            elif settings.vector_store_backend == "qdrant":
                report = build_qdrant_index(
                    document_dir=args.directory,
                    qdrant_path=settings.qdrant_path,
                    collection_name=settings.qdrant_collection,
                    embedder=embedder,
                    embedding_model=settings.embedding_model,
                    chunk_size=settings.chunk_size,
                    overlap=settings.chunk_overlap,
                )

            else:
                raise ValueError(
                    f"unsupported vector store backend: "
                    f"{settings.vector_store_backend}"
                )

            print(
                f"Indexed {report.document_count} documents "
                f"and {report.chunk_count} chunks"
            )

            for name, error in report.errors.items():
                print(f"Skipped {name}: {error}")

            return 0

        retriever = create_retriever(
            client=client,
            settings=settings,
        )

        if args.command == "ask":
            generator = AnswerGenerator(
                api=client.chat.completions,
                model=settings.chat_model
            )
            service = RAGService(
                retriever=retriever,
                generator=generator,
                evidence_policy=EvidencePolicy(
                    minimum_score=settings.evidence_minimum_score,
                    minimum_results=settings.evidence_minimum_results,
                ),
            )

            print_answer(service.ask(args.question))
            return 0

        agent = KnowledgeAgent(
            api=client.chat.completions,
            model=settings.chat_model,
            retriever=retriever,
        )

        while True:
            question = input("You> ").strip()

            if question.lower() in {
                "exit",
                "quit",
                "退出",
            }:
                return 0

            if question:
                print(f"Assistant> {agent.run(question)}")

    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#  python -m src.cli index .\documents
#  python -m src.cli ask "你的问题"
#  python -m src.cli chat
