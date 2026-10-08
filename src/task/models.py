"""Serializable models for generic long-running tasks."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskStep:
    id: str
    name: str
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    requires_approval: bool = False
    max_retries: int = 0
    retry_count: int = 0
    result: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.name:
            raise ValueError("step id and name are required")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.status = StepStatus(self.status)

    def transition(self, status: StepStatus) -> None:
        status = StepStatus(status)
        allowed = {
            StepStatus.PENDING: {StepStatus.READY, StepStatus.SKIPPED},
            StepStatus.READY: {StepStatus.RUNNING, StepStatus.SKIPPED},
            StepStatus.RUNNING: {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.WAITING_APPROVAL},
            StepStatus.WAITING_APPROVAL: {StepStatus.RUNNING, StepStatus.FAILED},
            StepStatus.FAILED: {StepStatus.READY, StepStatus.RUNNING},
            StepStatus.COMPLETED: set(),
            StepStatus.SKIPPED: set(),
        }
        if status != self.status and status not in allowed[self.status]:
            raise ValueError(f"invalid step transition: {self.status.value} -> {status.value}")
        self.status = status


@dataclass
class StepResult:
    step_id: str
    status: StepStatus
    result: Any = None
    error: str | None = None
    retryable: bool = False
    artifacts: dict[str, Any] = field(default_factory=dict)
    superstep_id: int = 0

    def __post_init__(self) -> None:
        self.status = StepStatus(self.status)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"status": self.status.value}


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    final_answer: Any = None
    state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
