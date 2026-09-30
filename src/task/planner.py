from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
from .models import TaskStep

class Planner(Protocol):
    def plan(self, goal: str) -> list[TaskStep]: ...

@dataclass
class StaticPlanner:
    steps: list[TaskStep]
    max_steps: int = 100
    available_tools: set[str] | None = None

    def plan(self, goal: str) -> list[TaskStep]:
        if not goal or not goal.strip(): raise ValueError("goal cannot be empty")
        if len(self.steps) > self.max_steps: raise ValueError("maximum step count exceeded")
        ids = {s.id for s in self.steps}
        if len(ids) != len(self.steps): raise ValueError("duplicate step id")
        for step in self.steps:
            missing = set(step.dependencies) - ids
            if missing: raise ValueError(f"unknown dependencies: {sorted(missing)}")
            if step.tool and self.available_tools is not None and step.tool not in self.available_tools:
                raise ValueError(f"unknown tool: {step.tool}")
        self._check_cycles()
        return [TaskStep(**{**s.__dict__, "dependencies": list(s.dependencies), "arguments": dict(s.arguments)}) for s in self.steps]

    def _check_cycles(self) -> None:
        graph = {s.id: s.dependencies for s in self.steps}
        visiting, visited = set(), set()
        def visit(node: str) -> None:
            if node in visiting: raise ValueError("cyclic step dependencies")
            if node in visited: return
            visiting.add(node)
            for dep in graph[node]: visit(dep)
            visiting.remove(node); visited.add(node)
        for node in graph: visit(node)

class ModelPlanner:
    def __init__(self, model: Any, *, max_steps: int = 100): self.model, self.max_steps = model, max_steps
    def plan(self, goal: str) -> list[TaskStep]:
        value = self.model(goal)
        if hasattr(value, "plan"): value = value.plan
        if not isinstance(value, list): raise ValueError("model planner must return a list")
        return StaticPlanner([s if isinstance(s, TaskStep) else TaskStep(**s) for s in value], self.max_steps).plan(goal)

