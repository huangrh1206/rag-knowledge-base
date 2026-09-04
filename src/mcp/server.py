"""Minimal allowlisted MCP server surface for internal tools."""

import json
from typing import Any

from src.agent.executor import ToolExecutor
from src.agent.tools import ToolRegistry
from src.mcp.models import MCPCapabilities, MCPToolDefinition


class LocalMCPServer:
    def __init__(
        self,
        registry: ToolRegistry,
        allowed_tools: set[str],
        executor: ToolExecutor | None = None,
    ) -> None:
        self._registry = registry
        self._allowed_tools = frozenset(allowed_tools)
        self._executor = executor or ToolExecutor(registry)

        missing = [
            name
            for name in self._allowed_tools
            if registry.get(name) is None
        ]
        if missing:
            raise ValueError(
                f"allowlisted tools are not registered: {', '.join(missing)}"
            )

    def initialize(self) -> MCPCapabilities:
        return MCPCapabilities(tools=True)

    def list_tools(self) -> list[MCPToolDefinition]:
        definitions: list[MCPToolDefinition] = []
        for name in sorted(self._allowed_tools):
            tool = self._registry.get(name)
            assert tool is not None
            function = tool.definition["function"]
            definitions.append(
                MCPToolDefinition(
                    name=name,
                    description=str(function.get("description", "")),
                    input_schema=dict(function.get("parameters", {})),
                )
            )
        return definitions

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        if name not in self._allowed_tools:
            return json.dumps(
                {"error": "tool is not exposed"},
                ensure_ascii=False,
            )
        return self._executor.invoke(
            name,
            json.dumps(arguments, ensure_ascii=False),
        )
