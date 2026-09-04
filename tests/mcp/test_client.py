import pytest

from src.mcp.client import MCPClient
from src.mcp.models import (
    MCPCapabilities,
    MCPPrompt,
    MCPResource,
    MCPToolDefinition,
)
from src.mcp.security import MCPServerPolicy


class FakeTransport:
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.capabilities = MCPCapabilities(
            tools=True,
            resources=True,
            prompts=True,
        )

    def initialize(self) -> MCPCapabilities:
        self.initialized = True
        return self.capabilities

    def list_tools(self) -> list[MCPToolDefinition]:
        return [MCPToolDefinition("lookup", "Look up a value", {})]

    def call_tool(self, name: str, arguments: dict) -> dict:
        return {"name": name, "arguments": arguments}

    def list_resources(self) -> list[MCPResource]:
        return [MCPResource("file://one", "One", "text/plain")]

    def list_prompts(self) -> list[MCPPrompt]:
        return [MCPPrompt("default", "Default prompt")]

    def close(self) -> None:
        self.closed = True


def test_client_exposes_transport_capabilities_and_operations() -> None:
    transport = FakeTransport()
    client = MCPClient(
        "docs",
        transport,
        MCPServerPolicy(allowed_server_ids={"docs"}),
    )

    with pytest.raises(RuntimeError):
        client.list_tools()

    assert client.initialize() == transport.capabilities
    assert client.list_tools()[0].name == "lookup"
    assert client.list_resources()[0].uri == "file://one"
    assert client.list_prompts()[0].name == "default"
    assert client.call_tool("lookup", {"term": "agent"})["name"] == "lookup"

    client.close()
    assert transport.closed is True


def test_client_rejects_unauthorized_server() -> None:
    with pytest.raises(PermissionError):
        MCPClient(
            "external",
            FakeTransport(),
            MCPServerPolicy(allowed_server_ids={"internal"}),
        )
