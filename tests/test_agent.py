from types import SimpleNamespace

import pytest

from src.agent import KnowledgeAgent
from src.agent_types import (
    AgentEmptyResponseError,
    AgentLimitError,
    AgentRunConfig,
    AgentStopReason,
    AgentValidationError,
)
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

    with pytest.raises(AgentLimitError, match="maximum rounds"):
        agent.run("Question")


def test_agent_returns_structured_result() -> None:
    response = SimpleNamespace(
        content="Direct answer",
        tool_calls=None,
    )
    agent = KnowledgeAgent(
        FakeCompletions([response]),
        "chat-model",
        FakeRetriever(),
    )

    result = agent.run_result("Question")

    assert result.answer == "Direct answer"
    assert result.rounds == 1
    assert result.tool_calls == 0
    assert result.stop_reason is AgentStopReason.COMPLETED


@pytest.mark.parametrize("question", ["", "   "])
def test_agent_rejects_blank_question(question: str) -> None:
    agent = KnowledgeAgent(
        FakeCompletions([]),
        "chat-model",
        FakeRetriever(),
    )

    with pytest.raises(AgentValidationError, match="question"):
        agent.run(question)


def test_agent_rejects_empty_model_response() -> None:
    response = SimpleNamespace(content="", tool_calls=None)
    agent = KnowledgeAgent(
        FakeCompletions([response]),
        "chat-model",
        FakeRetriever(),
    )

    with pytest.raises(AgentEmptyResponseError, match="empty content"):
        agent.run("Question")


def test_agent_stops_at_maximum_tool_calls() -> None:
    response = SimpleNamespace(
        content=None,
        tool_calls=[tool_call(), tool_call()],
    )
    agent = KnowledgeAgent(
        FakeCompletions([response]),
        "chat-model",
        FakeRetriever(),
        run_config=AgentRunConfig(max_tool_calls=1),
    )

    with pytest.raises(AgentLimitError, match="maximum tool calls"):
        agent.run("Question")


def test_agent_rejects_ambiguous_round_configuration() -> None:
    with pytest.raises(
        AgentValidationError,
        match="cannot both be provided",
    ):
        KnowledgeAgent(
            FakeCompletions([]),
            "chat-model",
            FakeRetriever(),
            max_rounds=2,
            run_config=AgentRunConfig(),
        )


def test_agent_rejects_non_positive_legacy_max_rounds() -> None:
    with pytest.raises(AgentValidationError, match="max rounds"):
        KnowledgeAgent(
            FakeCompletions([]),
            "chat-model",
            FakeRetriever(),
            max_rounds=0,
        )
