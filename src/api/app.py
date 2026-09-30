"""FastAPI adapter for the Agent runtime.

The application accepts an already constructed Agent-like object so transport
concerns stay separate from model, retrieval, and memory composition.
"""

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field


class AgentLike(Protocol):
    def run(self, question: str) -> str:
        ...


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)


class AskResponse(BaseModel):
    answer: str


def create_app(
    agent: AgentLike,
    *,
    health_check: Callable[[], dict[str, Any]] | None = None,
) -> Any:
    """Create an HTTP application around an injected Agent instance."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            "FastAPI API support requires the optional 'api' dependencies"
        ) from exc

    app = FastAPI(title="RAG Agent API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return health_check() if health_check is not None else {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            answer = agent.run(request.question)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="agent request failed",
            ) from exc
        return AskResponse(answer=answer)

    return app
