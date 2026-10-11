from __future__ import annotations
from typing import Any
import time
from langgraph.checkpoint.memory import InMemorySaver
from .state import TaskState
from .models import TaskResult, TaskStatus
from .planner import Planner, StaticPlanner
from .scheduler import SuperstepScheduler
from .policy import TaskPolicy
from .graph import build_task_graph
from .evaluator import CompletionEvaluator, Evaluation, Evaluator

class TaskRuntime:
    def __init__(self, *, planner: Planner | None = None, scheduler: SuperstepScheduler | None = None, checkpointer: Any | None = None, policy: TaskPolicy | None = None, evaluator: Evaluator | None = None):
        self.planner, self.scheduler, self.checkpointer, self.policy = planner, scheduler or SuperstepScheduler(), checkpointer or InMemorySaver(), policy or TaskPolicy()
        self.evaluator = evaluator or CompletionEvaluator()
        self.graph = build_task_graph(self.planner or StaticPlanner([]), self.scheduler, checkpointer=self.checkpointer, max_supersteps=self.policy.max_supersteps)
        self._states: dict[str, TaskState] = {}
        self._paused: set[str] = set()
        storage = getattr(self.checkpointer, "_task_states", {})
        if isinstance(storage, dict):
            for task_id, value in storage.items():
                if isinstance(value, dict) and "goal" in value:
                    self._states[task_id] = TaskState.from_dict(value)
    def start(self, task_id: str, goal: str, *, plan: list[Any] | None = None) -> TaskResult:
        if not task_id or not goal.strip(): raise ValueError("task_id and goal are required")
        if task_id in self._states: return self.resume(task_id)
        planner = self.planner or StaticPlanner(plan or [])
        state = TaskState(task_id, goal)
        state.plan = self._make_plan(planner, goal, state)
        state.status = TaskStatus.RUNNING
        self._states[task_id] = state
        return self._run(state)
    def resume(self, task_id: str, value: Any = None) -> TaskResult:
        if task_id not in self._states: raise KeyError(task_id)
        state = self._states[task_id];
        if state.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}: return self._result(state)
        if state.status == TaskStatus.WAITING_APPROVAL:
            decision = value
            approved = decision is True or (isinstance(decision, dict) and decision.get("action") in {"approve", "edit"})
            waiting = next((s for s in state.plan if s.status.name == "WAITING_APPROVAL"), None)
            if not approved:
                state.status, state.error = TaskStatus.FAILED, "human approval denied"
                if waiting: waiting.transition("failed"); waiting.error = state.error
                return self._result(state)
            if waiting:
                if isinstance(decision, dict) and decision.get("action") == "edit": waiting.arguments = dict(decision.get("arguments", waiting.arguments))
                waiting.transition("running")
        state.status = TaskStatus.RUNNING; self._paused.discard(task_id); return self._run(state)
    def pause(self, task_id: str) -> TaskResult:
        state = self._states[task_id]; self._paused.add(task_id); state.status = TaskStatus.PAUSED; state.stop_reason = {"code": "paused"}; return self._result(state)
    def cancel(self, task_id: str) -> TaskResult:
        state = self._states[task_id]; state.status = TaskStatus.CANCELLED; state.stop_reason = {"code": "cancelled"}; return self._result(state)
    def get_state(self, task_id: str) -> dict[str, Any]: return self._states[task_id].to_dict()
    def _run(self, state: TaskState) -> TaskResult:
        started = time.monotonic()
        while state.superstep_id < self.policy.max_supersteps and state.status == TaskStatus.RUNNING:
            if self.policy.task_timeout is not None and time.monotonic() - started >= self.policy.task_timeout:
                state.status = TaskStatus.FAILED
                state.error = "task timeout"
                state.stop_reason = {"code": "task_timeout", "timeout_seconds": self.policy.task_timeout}
                break
            if state.task_id in self._paused:
                state.status = TaskStatus.PAUSED; break
            waiting = next((s for s in state.plan if s.id in self.scheduler.find_ready(state) and s.requires_approval), None)
            if waiting:
                waiting.transition("ready") if waiting.status.value == "pending" else None
                waiting.transition("waiting_approval")
                state.current_step, state.status = waiting.id, TaskStatus.WAITING_APPROVAL
                state.events.append({"type": "approval_required", "step_id": waiting.id, "tool": waiting.tool})
                break
            if not state.ready_steps: state.ready_steps = self.scheduler.find_ready(state)
            if not state.ready_steps:
                state.status = TaskStatus.FAILED if state.failed_steps else TaskStatus.COMPLETED; break
            result = self.scheduler.run(state, state.ready_steps)
            self.scheduler.merge(state, result)
            state.ready_steps = self.scheduler.find_ready(state)
            evaluation = self.evaluator.evaluate(state.goal, state)
            state.events.append({
                "type": "evaluation",
                "superstep_id": state.superstep_id - 1,
                "sufficient": evaluation.sufficient,
                "feedback": evaluation.feedback,
            })
            if not evaluation.sufficient and self.planner is not None:
                proposed = self._make_plan(self.planner, state.goal, state, evaluation.constraints)
                if not self._apply_replan(state, proposed):
                    state.status = TaskStatus.FAILED
                    state.error = evaluation.feedback or "planner made no progress"
                    state.stop_reason = {"code": "planner_no_progress", "feedback": state.error}
                    break
                state.ready_steps = self.scheduler.find_ready(state)
            retryable = [s for s in state.plan if s.id in state.failed_steps and s.retry_count < max(s.max_retries, self.policy.max_retries)]
            for step in retryable:
                state.failed_steps.remove(step.id); step.retry_count += 1; step.transition("ready"); state.retry_count += 1
            if state.failed_steps: state.status = TaskStatus.FAILED; break
            self._checkpoint(state)
        if state.status == TaskStatus.RUNNING:
            state.status = TaskStatus.FAILED; state.error = "maximum supersteps exceeded"
            state.stop_reason = {"code": "max_supersteps", "limit": self.policy.max_supersteps}
        elif state.status == TaskStatus.FAILED and state.stop_reason is None:
            state.stop_reason = {"code": "step_failed", "steps": list(state.failed_steps), "error": state.error}
        if state.status == TaskStatus.COMPLETED: state.final_answer = state.artifacts
        self._checkpoint(state)
        return self._result(state)
    def _checkpoint(self, state: TaskState) -> None:
        self._states[state.task_id] = TaskState.from_dict(state.to_dict())
        storage = getattr(self.checkpointer, "_task_states", None)
        if storage is None:
            storage = {}
            setattr(self.checkpointer, "_task_states", storage)
        storage[state.task_id] = state.to_dict()
    def _result(self, state): return TaskResult(state.task_id, state.status, state.final_answer, state.to_dict(), state.error)

    def _make_plan(self, planner: Planner, goal: str, state: TaskState, constraints=None):
        definitions = []
        executor = getattr(self.scheduler, "executor", None)
        registry = getattr(executor, "registry", None)
        if registry is not None:
            definitions = registry.definitions()
        return planner.plan(
            goal,
            state=state.to_dict(),
            tools=definitions,
            constraints=constraints or {},
        )

    @staticmethod
    def _apply_replan(state: TaskState, proposed) -> bool:
        current = {step.id: step for step in state.plan}
        changed = False
        merged = []
        for step in proposed:
            existing = current.get(step.id)
            if existing is not None and existing.status.value == "completed":
                merged.append(existing)
                continue
            if existing is None or existing.__dict__ != step.__dict__:
                changed = True
            merged.append(step)
        if any(step.id not in {item.id for item in proposed} for step in state.plan if step.status.value == "completed"):
            merged = [step for step in state.plan if step.status.value == "completed"] + merged
        state.plan = merged
        state.events.append({"type": "replanned", "step_ids": [step.id for step in merged]})
        return changed or bool(merged)
