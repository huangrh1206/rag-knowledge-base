import time
from typing import Any

from src.agent.model_gateway import (
    AgentModelGateway,
    ChatCompletionsGateway,
)
from src.agent.tools import RAGSearchTool, ToolRegistry
from src.agent.executor import ToolExecutor
from src.agent.types import (
    AgentEmptyResponseError,
    AgentLimitError,
    AgentResult,
    AgentRunConfig,
    AgentStopReason,
    AgentValidationError,
)

AGENT_PROMPT = (
    "You are a technical knowledge-base agent. "
    "Use the knowledge search tool for document-grounded questions "
    "and cite its evidence."
)


class KnowledgeAgent:
    def __init__(
        self,
        api: Any,
        model: str,
        retriever: Any,
        max_rounds: int | None = None,
        *,
        run_config: AgentRunConfig | None = None,
        gateway: AgentModelGateway | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        if max_rounds is not None and run_config is not None:
            raise AgentValidationError(
                "max_rounds and run_config cannot both be provided"
            )

        self._gateway = gateway or ChatCompletionsGateway(api, model)
        self._retriever = retriever
        self._registry = registry or ToolRegistry(
            [RAGSearchTool(retriever)]
        )
        self._executor = ToolExecutor(self._registry)
        self._run_config = run_config or AgentRunConfig(
            max_rounds=(
                max_rounds
                if max_rounds is not None
                else 3
            )
        )

    def run(self, question: str) -> str:
        return self.run_result(question).answer

    def run_result(self, question: str) -> AgentResult:
        if not isinstance(question, str) or not question.strip():
            raise AgentValidationError(
                "question must be a non-empty string"
            )

        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": AGENT_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]
        started = time.monotonic()
        tool_call_count = 0

        for round_number in range(
            1,
            self._run_config.max_rounds + 1,
        ):
            self._check_elapsed(started)
            response = self._gateway.complete(
                messages=messages,
                tools=self._registry.definitions(),
            )
            self._check_elapsed(started)

            tool_calls = response.tool_calls
            assistant_message: dict[str, object] = {
                "role": "assistant",
                "content": response.content or "",
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(assistant_message)

            if not tool_calls:
                if not response.content or not response.content.strip():
                    raise AgentEmptyResponseError(
                        "agent returned empty content"
                    )
                return AgentResult(
                    answer=response.content.strip(),
                    rounds=round_number,
                    tool_calls=tool_call_count,
                    stop_reason=AgentStopReason.COMPLETED,
                )

            tool_call_count += len(tool_calls)
            if tool_call_count > self._run_config.max_tool_calls:
                raise AgentLimitError(
                    "agent exceeded maximum tool calls"
                )

            for call in tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": self._executor.invoke(
                            call.name,
                            call.arguments,
                            call_id=call.id,
                        ),
                    }
                )

        raise AgentLimitError(
            "agent exceeded maximum rounds"
        )

    def _check_elapsed(self, started: float) -> None:
        if time.monotonic() - started > self._run_config.max_elapsed_seconds:
            raise AgentLimitError(
                "agent exceeded maximum elapsed time"
            )
