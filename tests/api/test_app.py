import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api import create_app


class FakeAgent:
    def run(self, question: str) -> str:
        return f"answer: {question}"


def test_health_endpoint_returns_status() -> None:
    client = TestClient(create_app(FakeAgent()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_validates_and_calls_agent() -> None:
    client = TestClient(create_app(FakeAgent()))

    response = client.post("/ask", json={"question": "How?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "answer: How?"}


def test_ask_endpoint_rejects_empty_question() -> None:
    client = TestClient(create_app(FakeAgent()))

    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
