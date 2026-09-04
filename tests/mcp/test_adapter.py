import json

from src.agent.tools import ToolRegistry
from src.mcp.adapter import MCPRemoteTool, register_mcp_tools
from src.mcp.client import MCPClient
from src.mcp.models import MCPCapabilities, MCPToolDefinition
from src.mcp.security import MCPServerPolicy


class ToolTransport:
    def initialize(self) -> MCPCapabilities:
        return MCPCapabilities(tools=True)

    def list_tools(self) -> list[MCPToolDefinition]:
        return [MCPToolDefinition("search", "Remote search", {"type": "object"})]

    def call_tool(self, name: str, arguments: dict) -> dict:
        return {"items": [arguments["query"]]}

    def list_resources(self) -> list:
        return []

    def list_prompts(self) -> list:
        return []

    def close(self) -> None:
        return None


def test_remote_tools_are_namespaced_and_marked_untrusted() -> None:
    client = MCPClient(
        "remote",
        ToolTransport(),
        MCPServerPolicy(allowed_server_ids={"remote"}),
    )
    client.initialize()
    registry = ToolRegistry()
    register_mcp_tools(client, registry)

    tool = registry.get("remote.search")
    result = json.loads(tool.invoke(json.dumps({"query": "MCP"})))

    assert result["source"] == "remote"
    assert result["untrusted"] is True
    assert result["content"]["items"] == ["MCP"]


def test_remote_tool_rejects_unexpected_arguments() -> None:
    definition = MCPToolDefinition("search", "Search", {})
    client = MCPClient(
        "remote",
        ToolTransport(),
        MCPServerPolicy(allowed_server_ids={"remote"}),
    )
    tool = MCPRemoteTool(client, definition)

    result = json.loads(tool.invoke("[]"))
    assert "error" in result
