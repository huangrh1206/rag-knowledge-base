import argparse
import logging
from pathlib import Path

from src.models import Answer

from openai import OpenAI

from src.api_client import create_openai_client
from src.config import Settings
from src.embeddings import EmbeddingClient
from src.retriever import Retriever
from src.vector_store import VectorStore
from src.agent import KnowledgeAgent
from src.generator import AnswerGenerator
from src.rag_service import RAGService, build_index
from src.evidence_policy import EvidencePolicy

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

def _retriever(
    client: OpenAI,
    settings: Settings,
) -> Retriever:
    store = VectorStore.load(settings.index_dir)

    return Retriever(
        embedder=_embedder(client, settings),
        store=store,
        top_k=settings.top_k,
        threshold=settings.similarity_threshold,
    )

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    args = build_parser().parse_args()

    try:
        settings = Settings.from_env()
        client = _client(settings)

        if args.command == "index":
            report = build_index(
                document_dir=args.directory,
                index_dir=settings.index_dir,
                embedder=_embedder(client, settings),
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )

            print(
                f"Indexed {report.document_count} documents "
                f"and {report.chunk_count} chunks"
            )

            for name, error in report.errors.items():
                print(f"Skipped {name}: {error}")

            return 0

        retriever = _retriever(client, settings)

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
