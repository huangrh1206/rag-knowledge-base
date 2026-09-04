"""Validation and trust policy at the MCP boundary."""

import re
from dataclasses import dataclass

from src.mcp.models import MCPToolDefinition


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_identifier(value: str, label: str) -> None:
    if not value or not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{label} must contain only letters, numbers, _ or -"
        )


def validate_tool_definition(tool: MCPToolDefinition) -> None:
    validate_identifier(tool.name, "MCP tool name")
    if not isinstance(tool.input_schema, dict):
        raise ValueError("MCP tool input schema must be an object")
    if tool.input_schema.get("type", "object") != "object":
        raise ValueError("MCP tool input schema type must be object")


@dataclass(frozen=True)
class MCPServerPolicy:
    allowed_server_ids: frozenset[str]

    def authorize(self, server_id: str) -> None:
        if server_id not in self.allowed_server_ids:
            raise PermissionError(
                f"MCP server is not allowed: {server_id}"
            )
