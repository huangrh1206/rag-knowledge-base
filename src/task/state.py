"""Task state and non-destructive update helpers."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import copy

from .models import TaskStatus, TaskStep


@dataclass
class TaskState:
    task_id: str
    goal: str
    plan: list[TaskStep] = field(default_factory=list)
    current_step: str | None = None
    ready_steps: list[str] = field(default_factory=list)
    completed_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    error: str | None = None
    final_answer: Any = None
    superstep_id: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.status = TaskStatus(self.status)
        self.plan = [s if isinstance(s, TaskStep) else TaskStep(**s) for s in self.plan]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["plan"] = [asdict(s) | {"status": s.status.value} for s in self.plan]
        return copy.deepcopy(value)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskState":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: copy.deepcopy(v) for k, v in value.items() if k in allowed})

    def update(self, patch: dict[str, Any]) -> "TaskState":
        """Apply a patch while retaining fields absent from the patch."""
        for key, value in patch.items():
            if not hasattr(self, key):
                raise KeyError(key)
            if key == "status":
                value = TaskStatus(value)
            elif key == "plan":
                value = [s if isinstance(s, TaskStep) else TaskStep(**s) for s in value]
            setattr(self, key, value)
        return self


def merge_state(base: TaskState | dict[str, Any], patch: dict[str, Any]) -> TaskState:
    state = base if isinstance(base, TaskState) else TaskState.from_dict(base)
    return state.update(patch)
