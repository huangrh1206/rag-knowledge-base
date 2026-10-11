from src.task.amap import register_amap_tools
from src.task.mcp_http import StreamableHTTPMCPClient
from src.task.tools import TaskToolRegistry


class FakeMCP:
    def list_tools(self):
        return [
            {"name": "maps_weather", "description": "weather", "inputSchema": {"type": "object"}},
            {"name": "maps_direction_transit_integrated", "description": "transit", "inputSchema": {"type": "object"}},
        ]

    def call_tool(self, name, arguments):
        return {"remote_name": name, "arguments": arguments}


def test_register_amap_tools_discovers_and_aliases_remote_tools():
    registry = TaskToolRegistry()
    names = register_amap_tools(registry, url="https://example.test/mcp", api_key="key", client=FakeMCP())

    assert names == ["weather", "route_transit"]
    assert registry.names() == ["weather", "route_transit"]
    assert registry.get("weather").parameters == {"type": "object"}
    assert registry.get("weather").invoke({"city": "南京"})["remote_name"] == "maps_weather"


def test_streamable_http_client_adds_key_to_url():
    client = StreamableHTTPMCPClient("https://example.test/mcp?foo=bar", api_key="secret")
    assert "foo=bar" in client.url
    assert "key=secret" in client.url
