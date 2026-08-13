import json
from typing import Any
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
        max_rounds: int = 3,
    ) -> None:
        self._api = api
        self._model = model
        self._retriever = retriever
        self._max_rounds = max_rounds

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

        for _ in range(self._max_rounds):
            response = self._api.create(
                model=self._model,
                messages=messages,
                tools=TOOLS,
                temperature=0,
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            assistant_message: dict[str, object] = {
                "role": "assistant",
                "content": message.content or "",
            }

            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]

            messages.append(assistant_message)

            if not tool_calls:
                if not message.content:
                    raise ValueError(
                        "agent returned empty content"
                    )
                return message.content.strip()

            for call in tool_calls:
                if call.function.name == "search_knowledge_base":
                    content = self._tool_result(
                        call.function.arguments
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
        raise RuntimeError(
            "agent exceeded maximum rounds"
        )



