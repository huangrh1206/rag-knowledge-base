"""Reliable and policy-aware execution of registered Agent tools."""

from dataclasses import dataclass
import json
import time
from typing import Callable

from src.agent.tools import AgentTool, ToolRegistry


@dataclass(frozen=True)
class ToolExecutionPolicy:
    timeout_seconds: float = 10.0
    max_retries: int = 0
    idempotent: bool = False
    requires_approval: bool = False # 是否需要审批

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max retries must be non-negative")
        if self.max_retries and not self.idempotent:
            # 重试工具强制需要幂等
            raise ValueError(
                "retries require an idempotent tool"
            )


ApprovalCallback = Callable[[str, str], bool]


class ToolExecutionError(RuntimeError):
    """A registered tool could not be executed successfully."""


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        policies: dict[str, ToolExecutionPolicy] | None = None,
        approval: ApprovalCallback | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._registry = registry
        self._policies = policies or {}
        self._approval = approval
        self._clock = clock or time.monotonic
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
            return self._error("unknown tool")

        policy = self._policies.get(name, ToolExecutionPolicy())
        if policy.requires_approval:
            if self._approval is None or not self._approval(name, arguments):
                return self._error("tool approval denied")

        attempts = policy.max_retries + 1
        last_error: Exception | None = None
        for _ in range(attempts):
            started = self._clock()
            try:
                result = tool.invoke(arguments)
                elapsed = self._clock() - started
                if elapsed > policy.timeout_seconds:
                    raise TimeoutError(
                        f"tool exceeded timeout: {name}"
                    )
                if call_id:
                    self._completed[call_id] = result
                return result
            except Exception as exc:
                last_error = exc

        assert last_error is not None
        return self._error(str(last_error))

    @staticmethod
    def _error(message: str) -> str:
        return json.dumps(
            {"error": message},
            ensure_ascii=False,
        )
