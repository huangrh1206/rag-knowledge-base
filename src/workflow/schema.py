"""Validated, versioned workflow configuration for visual builders."""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any


CURRENT_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "callable",
        "credential",
        "password",
        "python_path",
        "secret",
        "token",
    }
)


class NodeKind(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    LLM = "llm"
    TOOL = "tool"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    APPROVAL = "approval"


@dataclass(frozen=True)
class WorkflowNode:
    node_id: str
    kind: NodeKind
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: object) -> "WorkflowNode":
        data = _require_dict(value, "workflow node")
        _reject_unknown(data, {"id", "kind", "config"}, "workflow node")
        node_id = _require_identifier(data.get("id"), "node id")
        try:
            kind = NodeKind(data.get("kind"))
        except ValueError as exc:
            raise ValueError(f"unsupported node kind: {data.get('kind')}") from exc
        config = _require_dict(data.get("config", {}), "node config")
        _validate_config(kind, config)
        return cls(node_id, kind, deepcopy(config))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "kind": self.kind.value,
            "config": deepcopy(self.config),
        }


@dataclass(frozen=True)
class WorkflowEdge:
    source: str
    target: str
    condition: str | None = None

    @classmethod
    def from_dict(cls, value: object) -> "WorkflowEdge":
        data = _require_dict(value, "workflow edge")
        _reject_unknown(data, {"source", "target", "condition"}, "workflow edge")
        condition = data.get("condition")
        if condition is not None and (
            not isinstance(condition, str) or not condition.strip()
        ):
            raise ValueError("edge condition must be a non-empty string")
        return cls(
            _require_identifier(data.get("source"), "edge source"),
            _require_identifier(data.get("target"), "edge target"),
            condition,
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"source": self.source, "target": self.target}
        if self.condition is not None:
            value["condition"] = self.condition
        return value


@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_id: str
    nodes: tuple[WorkflowNode, ...]
    edges: tuple[WorkflowEdge, ...]
    schema_version: int = CURRENT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: object) -> "WorkflowDefinition":
        data = migrate_workflow(_require_dict(value, "workflow"))
        _reject_unknown(
            data,
            {"schema_version", "workflow_id", "nodes", "edges", "metadata"},
            "workflow",
        )
        version = data.get("schema_version")
        if version != CURRENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported workflow schema version: {version}")
        workflow_id = _require_identifier(data.get("workflow_id"), "workflow id")
        node_values = data.get("nodes")
        edge_values = data.get("edges")
        if not isinstance(node_values, list) or not node_values:
            raise ValueError("workflow nodes must be a non-empty list")
        if not isinstance(edge_values, list):
            raise ValueError("workflow edges must be a list")
        nodes = tuple(WorkflowNode.from_dict(item) for item in node_values)
        edges = tuple(WorkflowEdge.from_dict(item) for item in edge_values)
        node_ids = [node.node_id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("workflow node ids must be unique")
        known_nodes = set(node_ids)
        for edge in edges:
            if edge.source not in known_nodes or edge.target not in known_nodes:
                raise ValueError("workflow edge references an unknown node")
        metadata = _require_dict(data.get("metadata", {}), "workflow metadata")
        return cls(workflow_id, nodes, edges, version, deepcopy(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workflow_id": self.workflow_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": deepcopy(self.metadata),
        }


def migrate_workflow(value: dict[str, Any]) -> dict[str, Any]:
    """Return a current-schema copy without mutating caller data."""

    migrated = deepcopy(value)
    version = migrated.get("schema_version", migrated.get("version", 0))
    if version == CURRENT_SCHEMA_VERSION:
        return migrated
    if version != 0:
        raise ValueError(f"unsupported workflow schema version: {version}")
    migrated["schema_version"] = CURRENT_SCHEMA_VERSION
    migrated["workflow_id"] = migrated.pop("id", migrated.get("workflow_id"))
    migrated.pop("version", None)
    for node in migrated.get("nodes", []):
        if isinstance(node, dict) and "kind" not in node and "type" in node:
            node["kind"] = node.pop("type")
    return migrated


def _validate_config(kind: NodeKind, config: dict[str, Any]) -> None:
    forbidden = _find_forbidden_key(config)
    if forbidden is not None:
        raise ValueError(f"node config cannot contain {forbidden}")
    if kind is NodeKind.TOOL:
        _require_identifier(config.get("tool_id"), "tool_id")
        version = config.get("tool_version")
        if version is not None and not isinstance(version, str):
            raise ValueError("tool_version must be a string")
    if kind is NodeKind.LOOP:
        maximum = config.get("max_iterations")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("loop max_iterations must be a positive integer")
    if kind is NodeKind.PARALLEL:
        branches = config.get("branches")
        if not isinstance(branches, list) or len(branches) < 2:
            raise ValueError("parallel branches must contain at least two entries")


def _find_forbidden_key(value: object) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_CONFIG_KEYS:
                return str(key)
            found = _find_forbidden_key(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_forbidden_key(nested)
            if found is not None:
                return found
    return None


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a valid identifier")
    return value


def _reject_unknown(
    value: dict[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(sorted(unknown))}")


WORKFLOW_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://example.local/schemas/workflow-v1.json",
    "type": "object",
    "required": ["schema_version", "workflow_id", "nodes", "edges"],
    "properties": {
        "schema_version": {"const": CURRENT_SCHEMA_VERSION},
        "workflow_id": {"type": "string", "pattern": _IDENTIFIER.pattern},
        "nodes": {"type": "array", "minItems": 1},
        "edges": {"type": "array"},
        "metadata": {"type": "object"},
    },
    "additionalProperties": False,
}
