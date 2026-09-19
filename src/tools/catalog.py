"""Versioned and tenant-aware metadata catalog for Agent tools."""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any

from src.agent.tools import AgentTool
from src.authz import AccessContext, ToolAccessPolicy


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class ToolSource(str, Enum):
    LOCAL = "local"
    INTERNAL_API = "internal_api"
    THIRD_PARTY_API = "third_party_api"
    MCP = "mcp"


class TrustLevel(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True)
class ToolDescriptor:
    tool_id: str
    version: str
    source: ToolSource
    trust: TrustLevel
    definition: dict[str, object]
    required_permissions: frozenset[str] = frozenset({"tools.invoke"})
    credential_ref: str | None = None

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.tool_id):
            raise ValueError("tool id must be a valid identifier")
        if not _SEMVER.fullmatch(self.version):
            raise ValueError("tool version must use MAJOR.MINOR.PATCH")
        if not isinstance(self.definition, dict):
            raise ValueError("tool definition must be an object")
        if self.credential_ref is not None and not _IDENTIFIER.fullmatch(
            self.credential_ref
        ):
            raise ValueError("credential reference must be a valid identifier")


@dataclass(frozen=True)
class CatalogEntry:
    descriptor: ToolDescriptor
    tool: AgentTool


@dataclass(frozen=True)
class ResolvedTool:
    descriptor: ToolDescriptor
    tool: AgentTool


class ToolCatalog:
    """Store tool metadata separately from the execution registry."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], CatalogEntry] = {}
        self._tenant_enabled: dict[tuple[str, str, str | None], bool] = {}

    def register(self, descriptor: ToolDescriptor, tool: AgentTool) -> None:
        key = (descriptor.tool_id, descriptor.version)
        if key in self._entries:
            raise ValueError(
                f"tool version already registered: {descriptor.tool_id}@{descriptor.version}"
            )
        function = descriptor.definition.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            raise ValueError("tool definition must contain a function name")
        self._entries[key] = CatalogEntry(descriptor, tool)

    def set_enabled(
        self,
        tenant_id: str,
        tool_id: str,
        enabled: bool,
        version: str | None = None,
    ) -> None:
        if not _IDENTIFIER.fullmatch(tenant_id):
            raise ValueError("tenant id must be a valid identifier")
        if not _IDENTIFIER.fullmatch(tool_id):
            raise ValueError("tool id must be a valid identifier")
        if version is not None and (tool_id, version) not in self._entries:
            raise KeyError(f"tool version is not registered: {tool_id}@{version}")
        self._tenant_enabled[(tenant_id, tool_id, version)] = enabled

    def resolve(
        self,
        tool_id: str,
        context: AccessContext,
        policy: ToolAccessPolicy,
        *,
        version: str | None = None,
    ) -> ResolvedTool:
        selected_version = version or self._latest_version(tool_id)
        entry = self._entries.get((tool_id, selected_version))
        if entry is None:
            raise KeyError(f"tool version is not registered: {tool_id}@{selected_version}")
        if not self._is_enabled(context.tenant_id, tool_id, selected_version):
            raise PermissionError(
                f"tool is disabled for tenant: {tool_id}@{selected_version}"
            )
        policy.authorize(context, entry.descriptor.required_permissions)
        return ResolvedTool(entry.descriptor, entry.tool)

    def descriptors(self) -> tuple[ToolDescriptor, ...]:
        entries = sorted(
            self._entries.values(),
            key=lambda item: (
                item.descriptor.tool_id,
                _version_key(item.descriptor.version),
            ),
        )
        return tuple(entry.descriptor for entry in entries)

    def _latest_version(self, tool_id: str) -> str:
        versions = [
            version
            for registered_id, version in self._entries
            if registered_id == tool_id
        ]
        if not versions:
            raise KeyError(f"tool is not registered: {tool_id}")
        return max(versions, key=_version_key)

    def _is_enabled(self, tenant_id: str, tool_id: str, version: str) -> bool:
        exact = (tenant_id, tool_id, version)
        general = (tenant_id, tool_id, None)
        if exact in self._tenant_enabled:
            return self._tenant_enabled[exact]
        return self._tenant_enabled.get(general, True)


def _version_key(version: str) -> tuple[int, int, int]:
    match = _SEMVER.fullmatch(version)
    if match is None:
        raise ValueError("tool version must use MAJOR.MINOR.PATCH")
    return tuple(int(part) for part in match.groups())
