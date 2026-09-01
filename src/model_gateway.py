from typing import Any, Protocol, Sequence

from src.agent_types import (
    AgentModelError,
    AgentModelResponse,
    AgentToolCall,
)


class AgentModelGateway(Protocol):
    def complete(
        self,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, object]],
    ) -> AgentModelResponse:
        ...


class ChatCompletionsGateway:
    """Normalize an OpenAI-compatible Chat Completions response."""

    def __init__(self, api: Any, model: str) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        self._api = api
        self._model = model

    def complete(
        self,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, object]],
    ) -> AgentModelResponse:
        try:
            response = self._api.create(
                model=self._model,
                messages=list(messages),
                tools=list(tools),
                temperature=0,
            )
            message = response.choices[0].message
            raw_tool_calls = message.tool_calls or []
            tool_calls = tuple(
                AgentToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in raw_tool_calls
            )
            return AgentModelResponse(
                content=message.content,
                tool_calls=tool_calls,
            )
        except AgentModelError:
            raise
        except Exception as exc:
            raise AgentModelError(
                "chat completions request failed"
            ) from exc
