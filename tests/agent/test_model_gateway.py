from types import SimpleNamespace

import pytest

from src.agent.types import AgentModelError
from src.agent.model_gateway import ChatCompletionsGateway


class RecordingAPI:
    def __init__(self, response: object) -> None:
        self.response = response
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.response


def test_gateway_normalizes_content_and_tool_calls() -> None:
    function = SimpleNamespace(
        name="search_knowledge_base",
        arguments='{"query":"FastAPI"}',
    )
    message = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(id="call-1", function=function)],
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )
    api = RecordingAPI(response)
    gateway = ChatCompletionsGateway(api, "chat-model")

    result = gateway.complete(
        messages=[{"role": "user", "content": "Question"}],
        tools=[],
    )

    assert result.content is None
    assert result.tool_calls[0].id == "call-1"
    assert result.tool_calls[0].name == "search_knowledge_base"
    assert api.kwargs["model"] == "chat-model"
    assert api.kwargs["temperature"] == 0


def test_gateway_wraps_malformed_responses() -> None:
    api = RecordingAPI(SimpleNamespace(choices=[]))
    gateway = ChatCompletionsGateway(api, "chat-model")

    with pytest.raises(
        AgentModelError,
        match="chat completions request failed",
    ):
        gateway.complete(messages=[], tools=[])


def test_gateway_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model cannot be empty"):
        ChatCompletionsGateway(RecordingAPI(object()), " ")
