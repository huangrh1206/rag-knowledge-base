"""Transport-neutral MCP capability models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MCPCapabilities:
    """描述 MCP 服务支持哪些能力"""
    tools: bool = False
    resources: bool = False
    prompts: bool = False


@dataclass(frozen=True)
class MCPToolDefinition:
    """描述 MCP 工具的定义"""
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPResource:
    """描述 MCP 资源，例如文件、文档、数据库记录"""
    uri: str
    name: str
    mime_type: str | None = None


@dataclass(frozen=True)
class MCPPrompt:
    name: str
    description: str = ""
