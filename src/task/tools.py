"""Tool registry and execution for the general task agent (RAG-free)."""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class GenericTool(Protocol):
    name: str
    description: str
    def invoke(self, arguments: dict[str, Any]) -> Any: ...


class ToolExecutionError(RuntimeError):
    def __init__(self, message: str, *, tool: str = "", retryable: bool = False):
        super().__init__(message)
        self.tool, self.retryable = tool, retryable


@dataclass
class ToolCallAudit:
    tool: str
    arguments: dict[str, Any]
    idempotency_key: str | None
    started_at: float
    duration: float
    success: bool
    error: str | None = None


class TaskToolRegistry:
    def __init__(self, tools: list[GenericTool] | None = None) -> None:
        self._tools: dict[str, GenericTool] = {}
        for tool in tools or []: self.register(tool)

    def register(self, tool: GenericTool) -> None:
        name = getattr(tool, "name", "")
        if not isinstance(name, str) or not name.strip(): raise ValueError("tool name cannot be empty")
        if name in self._tools: raise ValueError(f"tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> GenericTool:
        try: return self._tools[name]
        except KeyError as exc: raise ToolExecutionError(f"unknown tool: {name}", tool=name) from exc

    def validate(self, name: str, arguments: dict[str, Any]) -> None:
        tool = self.get(name)
        schema = getattr(tool, "parameters", None)
        if schema is None:
            definition = getattr(tool, "definition", {})
            schema = definition.get("function", {}).get("parameters") if isinstance(definition, dict) else None
        if not schema:
            return
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(arguments)
        extra = set(arguments) - set(properties)
        if missing: raise ToolExecutionError(f"missing arguments: {sorted(missing)}", tool=name)
        if extra and schema.get("additionalProperties") is False:
            raise ToolExecutionError(f"unexpected arguments: {sorted(extra)}", tool=name)

    def names(self) -> list[str]: return list(self._tools)
    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": getattr(t, "description", ""),
                "parameters": getattr(t, "parameters", None),
            }
            for t in self._tools.values()
        ]


class MCPToolAdapter:
    """Adapt an MCP client exposing ``call_tool(name, arguments)``."""
    def __init__(
        self,
        client: Any,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        *,
        remote_name: str | None = None,
    ) -> None:
        self.client = client
        self.name = name
        self.remote_name = remote_name or name
        self.description = description
        self.parameters = parameters
    def invoke(self, arguments: dict[str, Any]) -> Any:
        return self.client.call_tool(self.remote_name, arguments)


class TaskToolExecutor:
    def __init__(self, registry: TaskToolRegistry, *, timeout: float = 30.0, audit_log: list[ToolCallAudit] | None = None) -> None:
        if timeout <= 0: raise ValueError("timeout must be positive")
        self.registry, self.timeout = registry, timeout
        self.audit_log = audit_log if audit_log is not None else []
        self._completed: dict[str, Any] = {}

    async def invoke_async(self, name: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> Any:
        if not isinstance(arguments, dict): raise ToolExecutionError("arguments must be an object", tool=name)
        if idempotency_key and idempotency_key in self._completed: return self._completed[idempotency_key]
        self.registry.validate(name, arguments)
        tool = self.registry.get(name)
        started = time.time()
        try:
            if inspect.iscoroutinefunction(getattr(tool, "invoke", None)):
                value = await asyncio.wait_for(tool.invoke(arguments), self.timeout)
            else:
                value = await asyncio.wait_for(asyncio.to_thread(tool.invoke, arguments), self.timeout)
                if inspect.isawaitable(value): value = await asyncio.wait_for(value, self.timeout)
            if idempotency_key: self._completed[idempotency_key] = value
            self.audit_log.append(ToolCallAudit(name, arguments, idempotency_key, started, time.time()-started, True))
            return value
        except asyncio.TimeoutError as exc:
            error = ToolExecutionError(f"tool timed out: {name}", tool=name, retryable=True)
            self.audit_log.append(ToolCallAudit(name, arguments, idempotency_key, started, time.time()-started, False, str(error)))
            raise error from exc
        except Exception as exc:
            error = exc if isinstance(exc, ToolExecutionError) else ToolExecutionError(str(exc), tool=name, retryable=False)
            self.audit_log.append(ToolCallAudit(name, arguments, idempotency_key, started, time.time()-started, False, str(error)))
            raise error from exc

    def invoke(self, name: str, arguments: dict[str, Any], *, idempotency_key: str | None = None) -> Any:
        return asyncio.run(self.invoke_async(name, arguments, idempotency_key=idempotency_key))
