"""Tool contracts and registry for Agent tool calling."""

import json
from typing import Any, Protocol

from src.retrieval.dense import format_evidence


class AgentTool(Protocol):
    name: str
    definition: dict[str, object]

    def invoke(self, arguments: str) -> str:
        ...


class RAGSearchTool:
    name = "search_knowledge_base"
    definition = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Search technical documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
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

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    def invoke(self, arguments: str) -> str:
        try:
            value = json.loads(arguments)
            if not isinstance(value, dict):
                raise ValueError("arguments must be a JSON object")

            extra = set(value) - {"query", "top_k"}
            if extra:
                names = ", ".join(sorted(extra))
                raise ValueError(f"unexpected arguments: {names}")

            query = value.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")

            top_k = value.get("top_k", 5)
            if (
                isinstance(top_k, bool)
                or not isinstance(top_k, int)
                or not 1 <= top_k <= 10
            ):
                raise ValueError(
                    "top_k must be an integer between 1 and 10"
                )

            try:
                results = self._retriever.search(query, top_k)
            except TypeError:
                results = self._retriever.search(query)

            return format_evidence(results) or "no sufficient knowledge found"
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return json.dumps(
                {"error": str(exc)},
                ensure_ascii=False,
            )


class ToolRegistry:
    def __init__(self, tools: list[AgentTool] | None = None) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if not tool.name.strip():
            raise ValueError("tool name cannot be empty")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def definitions(self) -> list[dict[str, object]]:
        return [tool.definition for tool in self._tools.values()]

    def invoke(self, name: str, arguments: str) -> str:
        tool = self.get(name)
        if tool is None:
            return json.dumps(
                {"error": "unknown tool"},
                ensure_ascii=False,
            )
        return tool.invoke(arguments)
