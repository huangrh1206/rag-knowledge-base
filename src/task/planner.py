from __future__ import annotations
from dataclasses import dataclass
import inspect
import json
from typing import Any, Protocol
from .models import TaskStep

class Planner(Protocol):
    def plan(
        self,
        goal: str,
        state: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> list[TaskStep]: ...

@dataclass
class StaticPlanner:
    steps: list[TaskStep]
    max_steps: int = 100
    available_tools: set[str] | None = None

    def plan(self, goal: str, state=None, tools=None, constraints=None) -> list[TaskStep]:
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
    """Model-backed, scene-independent planner.

    The model only proposes a plan. Validation remains deterministic and local:
    unknown tools, malformed steps, cycles and excessive plans are rejected
    before a runtime can execute anything.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_steps: int = 100,
        available_tools: set[str] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        constraints: dict[str, Any] | None = None,
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.model = model
        self.max_steps = max_steps
        self.available_tools = available_tools
        self.tool_definitions = tool_definitions or []
        self.constraints = constraints or {}

    def plan(
        self,
        goal: str,
        state: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> list[TaskStep]:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal cannot be empty")
        definitions = tools if tools is not None else self.tool_definitions
        available = self.available_tools
        if available is None and definitions:
            available = {
                str(item.get("name") or item.get("function", {}).get("name"))
                for item in definitions
                if isinstance(item, dict)
            }
            available.discard("None")
        request = {
            "goal": goal,
            "state": state or {},
            "tools": definitions,
            "constraints": {**self.constraints, **(constraints or {})},
            "output_schema": self.output_schema(),
        }
        value = self._invoke(request)
        steps = self._parse(value)
        return StaticPlanner(
            steps,
            max_steps=self.max_steps,
            available_tools=available,
        ).plan(goal)

    @staticmethod
    def output_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["steps"],
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "name"],
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "tool": {"type": ["string", "null"]},
                            "arguments": {"type": "object"},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                            "requires_approval": {"type": "boolean"},
                            "max_retries": {"type": "integer", "minimum": 0},
                        },
                    },
                }
            },
        }

    def _invoke(self, request: dict[str, Any]) -> Any:
        if hasattr(self.model, "complete"):
            response = self.model.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a generic task planner. Return only JSON matching "
                            "the supplied output_schema. Never execute tools."
                        ),
                    },
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
                [],
            )
            return getattr(response, "content", response)
        try:
            value = self.model(request)
        except TypeError:
            # Keep compatibility with the original goal-only ModelPlanner API.
            value = self.model(request["goal"])
        if inspect.isawaitable(value):
            raise TypeError("ModelPlanner model must be synchronous; use an async adapter")
        return value

    @staticmethod
    def _parse(value: Any) -> list[TaskStep]:
        if hasattr(value, "plan"):
            value = value.plan
        if hasattr(value, "content"):
            value = value.content
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("model planner returned invalid JSON") from exc
        if isinstance(value, dict):
            value = value.get("steps", value.get("plan"))
        if not isinstance(value, list):
            raise ValueError("model planner must return a list or an object with steps")
        try:
            return [step if isinstance(step, TaskStep) else TaskStep(**step) for step in value]
        except (TypeError, ValueError) as exc:
            raise ValueError("model planner returned invalid task steps") from exc
