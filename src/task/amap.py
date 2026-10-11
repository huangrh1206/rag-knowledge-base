"""High-level registration of remote AMap MCP tools."""
from __future__ import annotations

from typing import Any

from .mcp_http import StreamableHTTPMCPClient
from .tools import MCPToolAdapter, TaskToolRegistry


AMAP_TOOL_ALIASES = {
    "maps_weather": "weather",
    "amap_weather": "weather",
    "maps_text_search": "poi_search",
    "amap_poi_search": "poi_search",
    "maps_direction_transit_integrated": "route_transit",
    "amap_direction_transit": "route_transit",
    "maps_direction_driving": "route_driving",
    "amap_direction_driving": "route_driving",
    "maps_direction_walking": "route_walking",
    "amap_direction_walking": "route_walking",
}


def register_amap_tools(
    registry: TaskToolRegistry,
    *,
    url: str,
    api_key: str,
    timeout: float = 30.0,
    client: StreamableHTTPMCPClient | None = None,
) -> list[str]:
    """Discover and register AMap tools from a Streamable HTTP MCP server."""
    mcp = client or StreamableHTTPMCPClient(url, api_key=api_key, timeout=timeout)
    registered: list[str] = []
    for definition in mcp.list_tools():
        remote_name = str(definition.get("name", "")).strip()
        if not remote_name:
            continue
        name = AMAP_TOOL_ALIASES.get(remote_name, remote_name)
        if name in registry.names():
            continue
        registry.register(
            MCPToolAdapter(
                mcp,
                name,
                str(definition.get("description", "AMap MCP tool")),
                definition.get("inputSchema") or definition.get("parameters"),
                remote_name=remote_name,
            )
        )
        registered.append(name)
    return registered
