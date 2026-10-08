from datetime import datetime

import argparse
import json
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
from src.conversation import ConversationContext
from src.memory.models import MemoryScope
from src.task.agent import GeneralTaskAgent
from src.task.models import TaskStep
from src.task.planner import StaticPlanner
from src.task.runtime import TaskRuntime
from src.task.scheduler import SuperstepScheduler
from src.task.tools import TaskToolExecutor, TaskToolRegistry
from src.task.chat import TaskChatResponder

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
    task_parser = subparsers.add_parser(
        "task",
        help="run a general long-running task without RAG retrieval",
    )
    task_parser.add_argument("goal")
    task_parser.add_argument(
        "--plan",
        help="JSON list of task steps; defaults to one value step",
    )
    task_chat_parser = subparsers.add_parser(
        "task-chat",
        help="start a multi-turn general agent chat without RAG retrieval",
    )
    task_chat_parser.add_argument("--user-id", default=None)
    task_chat_parser.add_argument("--session-id", default="default")
    return parser


def _create_task_agent(goal: str, plan_json: str | None) -> GeneralTaskAgent:
    if plan_json:
        raw_plan = json.loads(plan_json)
        if not isinstance(raw_plan, list):
            raise ValueError("--plan must be a JSON list")
        steps = [item if isinstance(item, TaskStep) else TaskStep(**item) for item in raw_plan]
    else:
        steps = [TaskStep("goal", "goal", arguments={"value": goal})]
    registry = GeneralTaskAgent().registry
    planner = StaticPlanner(steps, available_tools=set(registry.names()))
    runtime = TaskRuntime(
        planner=planner,
        scheduler=SuperstepScheduler(TaskToolExecutor(registry)),
    )
    return GeneralTaskAgent(runtime=runtime, registry=registry)


def _run_task_chat(
    settings: Settings,
    client: OpenAI,
    user_id: str | None = None,
    session_id: str = "default",
) -> int:
    registry = GeneralTaskAgent().registry
    responder = TaskChatResponder(client.chat.completions, settings.chat_model)
    conversation = ConversationContext.from_settings(
        MemoryScope(
            tenant_id=settings.task_tenant_id,
            user_id=user_id or settings.task_user_id,
            scene_id=settings.task_agent_id,
        ),
        settings=settings,
        session_id=session_id,
    )
    agent = GeneralTaskAgent(
        responder=responder,
        registry=registry,
        conversation=conversation,
    )
    while True:
        message = input("You> ").strip()
        if message.lower() in {"exit", "quit", "q"}:
            return 0
        if message.lower() == "reset":
            agent.reset_chat()
            print("Conversation memory reset.")
            continue
        if not message:
            continue
        result = agent.chat(message)
        print(f"Assistant> {result.final_answer}")

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


def _create_chat_agent(
    settings: Settings,
    client: OpenAI,
    retriever,
) -> KnowledgeAgent:
    conversation = ConversationContext.from_settings(
        MemoryScope(
            tenant_id="default",
            user_id="cli-user",
            scene_id="chat",
        ),
        settings=settings,
    )
    return KnowledgeAgent(
        api=client.chat.completions,
        model=settings.chat_model,
        retriever=retriever,
        conversation=conversation,
    )


def main() -> int:
    configure_logging()

    args = build_parser().parse_args()

    try:
        if args.command == "task":
            result = _create_task_agent(args.goal, args.plan).run(args.goal)
            print(json.dumps({
                "task_id": result.task_id,
                "status": result.status.value,
                "final_answer": result.final_answer,
                "error": result.error,
            }, ensure_ascii=False, default=str))
            return 0 if result.status.value == "completed" else 1

        settings = Settings.from_env()
        client = _client(settings)

        if args.command == "task-chat":
            return _run_task_chat(
                settings,
                client,
                args.user_id,
                args.session_id,
            )

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

        agent = _create_chat_agent(settings, client, retriever)

        while True:
            question = input("You> ").strip()

            if question.lower() in {
                "exit",
                "quit",
                "退出",
            }:
                return 0

            if question.lower() == "reset":
                agent.reset_conversation()
                print("Conversation memory reset.")
                continue

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
#  python -m src.cli task-chat --session-id trip
