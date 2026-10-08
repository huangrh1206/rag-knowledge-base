from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any
from .models import StepResult, StepStatus, TaskStep
from .state import TaskState
from .tools import TaskToolExecutor

@dataclass
class SuperstepResult:
    superstep_id: int
    results: list[StepResult]
    ready_steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"superstep_id": self.superstep_id, "results": [r.to_dict() for r in self.results], "ready_steps": list(self.ready_steps)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SuperstepResult":
        return cls(value["superstep_id"], [StepResult(**r) for r in value.get("results", [])], value.get("ready_steps", []))

class SuperstepScheduler:
    def __init__(self, executor: TaskToolExecutor | None = None, *, parallel: bool = True): self.executor, self.parallel = executor, parallel
    def find_ready(self, state: TaskState) -> list[str]:
        done = set(state.completed_steps)
        return [s.id for s in state.plan if s.status in {StepStatus.PENDING, StepStatus.READY, StepStatus.FAILED} and set(s.dependencies) <= done and s.id not in state.completed_steps]
    async def run_async(self, state: TaskState, step_ids: list[str] | None = None) -> SuperstepResult:
        ids = step_ids if step_ids is not None else self.find_ready(state)
        steps = [next(s for s in state.plan if s.id == i) for i in ids]
        async def one(step: TaskStep) -> StepResult:
            try:
                if self.executor is None or step.tool is None: value = step.arguments.get("value", step.arguments)
                else: value = await self.executor.invoke_async(step.tool, step.arguments, idempotency_key=f"{state.task_id}:{step.id}")
                return StepResult(step.id, StepStatus.COMPLETED, value, superstep_id=state.superstep_id)
            except Exception as exc:
                return StepResult(step.id, StepStatus.FAILED, error=str(exc), retryable=getattr(exc, "retryable", False), superstep_id=state.superstep_id)
        results = await asyncio.gather(*(one(s) for s in steps)) if self.parallel else [await one(s) for s in steps]
        return SuperstepResult(state.superstep_id, results, [])
    def run(self, state: TaskState, step_ids: list[str] | None = None) -> SuperstepResult:
        return asyncio.run(self.run_async(state, step_ids))
    def merge(self, state: TaskState, result: SuperstepResult) -> TaskState:
        state.events.append({"type": "superstep_started", "superstep_id": result.superstep_id, "steps": [r.step_id for r in result.results]})
        for item in result.results:
            step = next(s for s in state.plan if s.id == item.step_id)
            step.status = item.status; step.result, step.error = item.result, item.error
            if item.status == StepStatus.COMPLETED:
                if item.step_id not in state.completed_steps: state.completed_steps.append(item.step_id)
                state.artifacts[item.step_id] = item.result
            else:
                if item.step_id not in state.failed_steps: state.failed_steps.append(item.step_id)
                state.error = item.error
            state.events.append({"type": "step_succeeded" if item.status == StepStatus.COMPLETED else "step_failed", "step_id": item.step_id, "superstep_id": result.superstep_id, "error": item.error})
        state.superstep_id += 1
        state.ready_steps = self.find_ready(state)
        return state
