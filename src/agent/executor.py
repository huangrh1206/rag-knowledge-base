"""Reliable and policy-aware execution of registered Agent tools."""

from dataclasses import dataclass
import json
import time
from typing import Any, Callable

from src.agent.tools import ToolRegistry


@dataclass(frozen=True)
class ToolExecutionPolicy:
    timeout_seconds: float = 10.0
    max_retries: int = 0
    idempotent: bool = False
    requires_approval: bool = False

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max retries must be non-negative")
        if self.max_retries and not self.idempotent:
            raise ValueError("retries require an idempotent tool")


ApprovalCallback = Callable[[str, str], bool]
EventCallback = Callable[[str, dict[str, Any]], None]


class ToolExecutionError(RuntimeError):
    """A registered tool could not be executed successfully."""


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policies: dict[str, ToolExecutionPolicy] | None = None,
        approval: ApprovalCallback | None = None,
        clock: Callable[[], float] | None = None,
        event_callback: EventCallback | None = None,
    ) -> None:
        self._registry = registry
        self._policies = policies or {}
        self._approval = approval
        self._clock = clock or time.monotonic
        self._event_callback = event_callback
        self._completed: dict[str, str] = {}

    def invoke(
        self,
        name: str,
        arguments: str,
        call_id: str | None = None,
    ) -> str:
        if call_id and call_id in self._completed:
            return self._completed[call_id]

        tool = self._registry.get(name)
        if tool is None:
            self._emit(
                "tool_failed",
                {"name": name, "error": "unknown tool", "call_id": call_id},
            )
            return self._error("unknown tool")

        policy = self._policies.get(name, ToolExecutionPolicy())
        if policy.requires_approval:
            if self._approval is None or not self._approval(name, arguments):
                self._emit(
                    "tool_failed",
                    {
                        "name": name,
                        "error": "tool approval denied",
                        "call_id": call_id,
                    },
                )
                return self._error("tool approval denied")

        self._emit(
            "tool_requested",
            {"name": name, "arguments": arguments, "call_id": call_id},
        )
        attempts = policy.max_retries + 1
        last_error: Exception | None = None
        for _ in range(attempts):
            started = self._clock()
            try:
                result = tool.invoke(arguments)
                elapsed = self._clock() - started
                if elapsed > policy.timeout_seconds:
                    raise TimeoutError(f"tool exceeded timeout: {name}")
                if call_id:
                    self._completed[call_id] = result
                self._emit(
                    "tool_completed",
                    {"name": name, "result": result, "call_id": call_id},
                )
                return result
            except Exception as exc:
                last_error = exc

        assert last_error is not None
        self._emit(
            "tool_failed",
            {"name": name, "error": str(last_error), "call_id": call_id},
        )
        return self._error(str(last_error))

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_callback is not None:
            self._event_callback(event_type, payload)

    @staticmethod
    def _error(message: str) -> str:
        return json.dumps({"error": message}, ensure_ascii=False)
