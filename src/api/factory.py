"""Production composition for the HTTP Agent application."""

from src.agent import KnowledgeAgent
from src.config import Settings
from src.conversation import ConversationContext
from src.infrastructure.openai_client import create_openai_client
from src.memory.models import MemoryScope
from src.retrieval.factory import create_retriever

from src.api.app import create_app


def create_agent(
    settings: Settings,
    *,
    scope: MemoryScope | None = None,
) -> KnowledgeAgent:
    """Build the configured Agent without duplicating CLI runtime logic."""

    client = create_openai_client(settings)
    retriever = create_retriever(client=client, settings=settings)
    conversation = None
    if scope is not None:
        conversation = ConversationContext.from_settings(
            scope,
            settings=settings,
        )
    return KnowledgeAgent(
        api=client.chat.completions,
        model=settings.chat_model,
        retriever=retriever,
        conversation=conversation,
    )


def create_production_app(
    settings: Settings | None = None,
    *,
    scope: MemoryScope | None = None,
):
    """Create the FastAPI application using environment-backed settings."""

    resolved = settings or Settings.from_env()
    return create_app(create_agent(resolved, scope=scope))
