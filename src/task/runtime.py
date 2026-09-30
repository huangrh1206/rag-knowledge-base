from __future__ import annotations
from typing import Any
try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # optional dependency in lightweight unit-test environments
    class InMemorySaver:
        def __init__(self): self.storage = {}
from .state import TaskState
from .models import TaskResult, TaskStatus
from .planner import Planner, StaticPlanner
from .scheduler import SuperstepScheduler
from .policy import TaskPolicy

class TaskRuntime:
    def __init__(self, *, planner: Planner | None = None, scheduler: SuperstepScheduler | None = None, checkpointer: Any | None = None, policy: TaskPolicy | None = None):
        self.planner, self.scheduler, self.checkpointer, self.policy = planner, scheduler or SuperstepScheduler(), checkpointer or InMemorySaver(), policy or TaskPolicy()
        self._states: dict[str, TaskState] = {}
    def start(self, task_id: str, goal: str, *, plan: list[Any] | None = None) -> TaskResult:
        if task_id in self._states: return self.resume(task_id)
        planner = self.planner or StaticPlanner(plan or [])
        state = TaskState(task_id, goal); state.plan = planner.plan(goal); state.status = TaskStatus.RUNNING; self._states[task_id] = state
        return self._run(state)
    def resume(self, task_id: str, value: Any = None) -> TaskResult:
        if task_id not in self._states: raise KeyError(task_id)
        state = self._states[task_id];
        if state.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}: return self._result(state)
        state.status = TaskStatus.RUNNING; return self._run(state)
    def pause(self, task_id: str) -> TaskResult:
        state = self._states[task_id]; state.status = TaskStatus.PAUSED; return self._result(state)
    def cancel(self, task_id: str) -> TaskResult:
        state = self._states[task_id]; state.status = TaskStatus.CANCELLED; return self._result(state)
    def get_state(self, task_id: str) -> dict[str, Any]: return self._states[task_id].to_dict()
    def _run(self, state: TaskState) -> TaskResult:
        while state.superstep_id < self.policy.max_supersteps and state.status == TaskStatus.RUNNING:
            if not state.ready_steps: state.ready_steps = self.scheduler.find_ready(state)
            if not state.ready_steps:
                state.status = TaskStatus.FAILED if state.failed_steps else TaskStatus.COMPLETED; break
            self.scheduler.merge(state, self.scheduler.run(state, state.ready_steps)); state.ready_steps = self.scheduler.find_ready(state)
            if state.failed_steps: state.status = TaskStatus.FAILED; break
        if state.status == TaskStatus.RUNNING: state.status = TaskStatus.FAILED; state.error = "maximum supersteps exceeded"
        if state.status == TaskStatus.COMPLETED: state.final_answer = state.artifacts
        return self._result(state)
    def _result(self, state): return TaskResult(state.task_id, state.status, state.final_answer, state.to_dict(), state.error)
