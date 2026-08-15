from types import SimpleNamespace

import pytest

from src.agent import KnowledgeAgent
from src.models import Chunk, SearchResult


def tool_call(
    arguments: str = '{"query":"FastAPI parameters","top_k":3}',
) -> SimpleNamespace:
    function = SimpleNamespace(
        name="search_knowledge_base",
        arguments=arguments,
    )
    return SimpleNamespace(id="call-1", function=function)


class FakeCompletions:
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self.messages = messages
        self.calls = 0

    def create(self, **kwargs: object) -> SimpleNamespace:
        message = self.messages[self.calls]
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


class FakeRetriever:
    def search(self, question: str) -> list[SearchResult]:
        chunk = Chunk(
            "guide-0000",
            "Use type annotations",
            "guide.docx",
            2,
            2,
        )
        return [SearchResult(chunk, 0.9)]


def test_agent_executes_search_tool_then_returns_answer() -> None:
    first = SimpleNamespace(
        content=None,
        tool_calls=[tool_call()],
    )
    second = SimpleNamespace(
        content="Use type annotations [1]",
        tool_calls=None,
    )
    api = FakeCompletions([first, second])
    agent = KnowledgeAgent(
        api,
        "chat-model",
        FakeRetriever(),
        max_rounds=3,
    )

    answer = agent.run("How does FastAPI declare parameters?")

    assert answer == "Use type annotations [1]"
    assert api.calls == 2


def test_agent_rejects_invalid_tool_arguments() -> None:
    first = SimpleNamespace(
        content=None,
        tool_calls=[tool_call("not-json")],
    )
    second = SimpleNamespace(
        content="Tool arguments are invalid",
        tool_calls=None,
    )
    agent = KnowledgeAgent(
        FakeCompletions([first, second]),
        "chat-model",
        FakeRetriever(),
    )

    assert agent.run("Question") == "Tool arguments are invalid"


def test_agent_stops_at_max_rounds() -> None:
    repeated = SimpleNamespace(
        content=None,
        tool_calls=[tool_call()],
    )
    agent = KnowledgeAgent(
        FakeCompletions([repeated, repeated]),
        "chat-model",
        FakeRetriever(),
        max_rounds=2,
    )

    with pytest.raises(RuntimeError, match="maximum rounds"):
        agent.run("Question")
