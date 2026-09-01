from types import SimpleNamespace

import pytest

from src.rag.generator import INSUFFICIENT_EVIDENCE, AnswerGenerator
from src.rag.models import Chunk, SearchResult


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        message = SimpleNamespace(content="FastAPI uses type annotations [1]")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


def test_generator_skips_api_when_evidence_is_empty() -> None:
    api = FakeCompletions()
    generator = AnswerGenerator(api, model="chat-model")

    answer = generator.generate("How do I declare parameters?", [])

    assert answer == INSUFFICIENT_EVIDENCE
    assert api.calls == []


def test_generator_sends_numbered_evidence_to_chat_api() -> None:
    api = FakeCompletions()
    generator = AnswerGenerator(api, model="chat-model")
    result = SearchResult(
        Chunk(
            "guide-0000",
            "Use type annotations",
            "guide.docx",
            2,
            2,
        ),
        0.9,
    )

    answer = generator.generate(
        "How do I declare parameters?",
        [result],
    )

    assert answer == "FastAPI uses type annotations [1]"
    call = api.calls[0]
    assert call["model"] == "chat-model"
    assert call["temperature"] == 0
    messages = call["messages"]
    assert "[1] 来源：guide.docx" in messages[1]["content"]
    assert "How do I declare parameters?" in messages[1]["content"]


def test_generator_rejects_empty_model_content() -> None:
    class EmptyCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            message = SimpleNamespace(content=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)]
            )

    generator = AnswerGenerator(
        EmptyCompletions(),
        model="chat-model",
    )
    result = SearchResult(
        Chunk("guide-0000", "Evidence", "guide.docx", 1, 1),
        0.9,
    )

    with pytest.raises(ValueError, match="chat model returned empty content"):
        generator.generate("Question", [result])
