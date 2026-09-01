"""Stable request, response, limit, and error types for agents."""

from dataclasses import dataclass
from enum import StrEnum


class AgentError(Exception):
    """Base error for failures at the agent runtime boundary."""


class AgentValidationError(AgentError, ValueError):
    """The caller supplied an invalid agent request or configuration."""


class AgentModelError(AgentError):
    """The model gateway failed to produce a usable response."""


class AgentEmptyResponseError(AgentModelError):
    """The model returned neither content nor tool calls."""


class AgentLimitError(AgentError, RuntimeError):
    """The run exceeded a configured safety limit."""


class AgentStopReason(StrEnum):
    COMPLETED = "completed"


@dataclass(frozen=True)
class AgentRunConfig:
    max_rounds: int = 3
    max_tool_calls: int = 10
    max_elapsed_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_rounds <= 0:
            raise AgentValidationError(
                "max rounds must be positive"
            )
        if self.max_tool_calls <= 0:
            raise AgentValidationError(
                "max tool calls must be positive"
            )
        if self.max_elapsed_seconds <= 0:
            raise AgentValidationError(
                "max elapsed seconds must be positive"
            )


@dataclass(frozen=True)
class AgentToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AgentModelResponse:
    content: str | None
    tool_calls: tuple[AgentToolCall, ...] = ()


@dataclass(frozen=True)
class AgentResult:
    answer: str
    rounds: int
    tool_calls: int
    stop_reason: AgentStopReason
