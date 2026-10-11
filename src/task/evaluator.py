"""Evaluation contracts for closed-loop task execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .state import TaskState


@dataclass(frozen=True)
class Evaluation:
    sufficient: bool
    feedback: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)


class Evaluator(Protocol):
    def evaluate(self, goal: str, state: TaskState) -> Evaluation: ...


class CompletionEvaluator:
    """Default evaluator: complete when no runnable or failed work remains."""

    def evaluate(self, goal: str, state: TaskState) -> Evaluation:
        sufficient = not state.ready_steps and not state.failed_steps
        return Evaluation(sufficient=sufficient)
