import json
import time
from typing import Any

from src.agent_types import (
    AgentEmptyResponseError,
    AgentLimitError,
    AgentResult,
    AgentRunConfig,
    AgentStopReason,
    AgentValidationError,
)
from src.model_gateway import (
    AgentModelGateway,
    ChatCompletionsGateway,
)
from src.retriever import Retriever, format_evidence

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search technical Word documents before answering "
                "document-grounded questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]

AGENT_PROMPT = """你是技术知识库 Agent。
寒暄和对话控制可以直接回答。
凡是涉及技术文档事实的问题，必须先调用 search_knowledge_base。
最终答案只能使用工具返回的资料，并保留资料编号引用；
资料不足时明确说明。"""


class KnowledgeAgent:
    def __init__(
        self,
        api: Any,
        model: str,
        retriever: Retriever,
        max_rounds: int | None = None,
        *,
        run_config: AgentRunConfig | None = None,
        gateway: AgentModelGateway | None = None,
    ) -> None:
        if max_rounds is not None and run_config is not None:
            raise AgentValidationError(
                "max_rounds and run_config cannot both be provided"
            )

        self._gateway = gateway or ChatCompletionsGateway(
            api,
            model,
        )
        self._retriever = retriever
        self._run_config = run_config or AgentRunConfig(
            max_rounds=(
                max_rounds
                if max_rounds is not None
                else 3
            ),
        )

    def _tool_result(self, arguments: str) -> str:
        try:
            value = json.loads(arguments)

            query = value["query"]
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    "query must be a non-empty string"
                )

            results = self._retriever.search(query)

            return (
                format_evidence(results)
                or "知识库中没有足够信息。"
            )
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            return json.dumps(
                {"error": str(exc)},
                ensure_ascii = False,
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
                tools=TOOLS,
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
                if call.name == "search_knowledge_base":
                    content = self._tool_result(
                        call.arguments
                    )
                else:
                    content = json.dumps(
                        {"error": "unknown tool"},
                        ensure_ascii=False,
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": content,
                    }
                )
        raise AgentLimitError(
            "agent exceeded maximum rounds"
        )

    def _check_elapsed(self, started: float) -> None:
        elapsed = time.monotonic() - started
        if elapsed > self._run_config.max_elapsed_seconds:
            raise AgentLimitError(
                "agent exceeded maximum elapsed time"
            )



