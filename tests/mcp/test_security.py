import pytest

from src.mcp.models import MCPToolDefinition
from src.mcp.security import (
    MCPServerPolicy,
    validate_identifier,
    validate_tool_definition,
)


def test_security_validates_identifiers_and_tool_schema() -> None:
    assert validate_identifier("server_1", "server id") is None
    validate_tool_definition(MCPToolDefinition("lookup", "Lookup", {}))

    with pytest.raises(ValueError):
        validate_identifier("server.with.dot", "server id")
    with pytest.raises(ValueError):
        validate_tool_definition(MCPToolDefinition("bad name", "", {}))


def test_policy_can_allow_all_or_selected_servers() -> None:
    MCPServerPolicy(allowed_server_ids={"any"}).authorize("any")
    with pytest.raises(PermissionError):
        MCPServerPolicy(allowed_server_ids={"known"}).authorize("unknown")
