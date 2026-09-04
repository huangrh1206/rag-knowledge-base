"""Protocol implemented by official-SDK and fake MCP transports."""

from typing import Any, Protocol

from src.mcp.models import (
    MCPCapabilities,
    MCPPrompt,
    MCPResource,
    MCPToolDefinition,
)


class MCPTransport(Protocol):
    def initialize(self) -> MCPCapabilities:
        ...

    def list_tools(self) -> list[MCPToolDefinition]:
        ...

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        ...

    def list_resources(self) -> list[MCPResource]:
        ...

    def list_prompts(self) -> list[MCPPrompt]:
        ...

    def close(self) -> None:
        ...
