from src.agent.tools import ToolRegistry
import json
from src.mcp.models import MCPCapabilities
from src.mcp.server import LocalMCPServer


class EchoTool:
    name = "echo"
    definition = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text",
            "parameters": {"type": "object"},
        },
    }

    def invoke(self, arguments: dict) -> str:
        value = json.loads(arguments)
        return value["text"]


def test_local_server_exposes_only_allowlisted_tools() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    server = LocalMCPServer(registry, allowed_tools={"echo"})

    assert server.initialize() == MCPCapabilities(tools=True)
    assert [item.name for item in server.list_tools()] == ["echo"]
    assert server.call_tool("echo", {"text": "ok"}) == "ok"
    assert json.loads(server.call_tool("missing", {}))["error"] == (
        "tool is not exposed"
    )
