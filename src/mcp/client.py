"""Lifecycle-aware MCP client independent of a concrete transport."""

from typing import Any

from src.mcp.models import (
    MCPCapabilities,
    MCPPrompt,
    MCPResource,
    MCPToolDefinition,
)
from src.mcp.security import (
    MCPServerPolicy,
    validate_identifier,
    validate_tool_definition,
)
from src.mcp.transport import MCPTransport


class MCPClient:
    """负责与一个 MCP 服务交互"""
    def __init__(
        self,
        server_id: str,
        transport: MCPTransport,
        policy: MCPServerPolicy,
    ) -> None:
        validate_identifier(server_id, "MCP server id")
        policy.authorize(server_id)
        self.server_id = server_id
        self._transport = transport
        self._capabilities: MCPCapabilities | None = None

    @property
    def capabilities(self) -> MCPCapabilities:
        if self._capabilities is None:
            raise RuntimeError("MCP client is not initialized")
        return self._capabilities

    def initialize(self) -> MCPCapabilities:
        self._capabilities = self._transport.initialize()
        return self._capabilities

    def list_tools(self) -> list[MCPToolDefinition]:
        if not self.capabilities.tools:
            return []
        tools = self._transport.list_tools()
        for tool in tools:
            validate_tool_definition(tool)
        return tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        if not self.capabilities.tools:
            raise RuntimeError("MCP server does not support tools")
        return self._transport.call_tool(name, arguments)

    def list_resources(self) -> list[MCPResource]:
        if not self.capabilities.resources:
            return []
        return self._transport.list_resources()

    def list_prompts(self) -> list[MCPPrompt]:
        if not self.capabilities.prompts:
            return []
        return self._transport.list_prompts()

    def close(self) -> None:
        self._transport.close()
