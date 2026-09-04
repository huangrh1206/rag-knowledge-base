"""Adapt discovered MCP tools to the internal AgentTool contract."""

import json

from src.agent.tools import ToolRegistry
from src.mcp.client import MCPClient
from src.mcp.models import MCPToolDefinition


class MCPRemoteTool:
    def __init__(
        self,
        client: MCPClient,
        remote: MCPToolDefinition,
    ) -> None:
        self._client = client
        self._remote = remote
        self.name = f"{client.server_id}.{remote.name}"
        self.definition = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": remote.description,
                "parameters": remote.input_schema,
            },
        }

    def invoke(self, arguments: str) -> str:
        try:
            value = json.loads(arguments)
            if not isinstance(value, dict):
                raise ValueError("arguments must be a JSON object")
            content = self._client.call_tool(
                self._remote.name,
                value,
            )
            return json.dumps(
                {
                    "source": self._client.server_id,
                    "untrusted": True, # 提醒上层这是外部不可信服务返回的数据，需要额外校验
                    "content": content,
                },
                ensure_ascii=False,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return json.dumps(
                {"error": str(exc)},
                ensure_ascii=False,
            )


def register_mcp_tools(
    client: MCPClient,
    registry: ToolRegistry,
) -> list[str]:
    names: list[str] = []
    for definition in client.list_tools():
        tool = MCPRemoteTool(client, definition)
        registry.register(tool)
        names.append(tool.name)
    return names
