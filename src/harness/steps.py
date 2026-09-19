"""Reusable steps for connecting Harness state to Agent tools."""

import json
from collections.abc import Callable
from typing import Any, Protocol

from langgraph.types import interrupt

from src.agent.executor import ToolExecutor

State = dict[str, Any]
ArgumentsBuilder = Callable[[State], dict[str, Any]]
StepHandler = Callable[[State], State]


class AgentRunner(Protocol):
    def run_result(self, question: str) -> Any:
        ...


class FunctionStep:
    """Give a named business function the Harness step contract."""

    def __init__(self, name: str, handler: StepHandler) -> None:
        if not name.strip():
            raise ValueError("step name cannot be empty")
        self.name = name
        self._handler = handler

    def __call__(self, state: State) -> State:
        return self._handler(dict(state))


class ToolStep:
    """Build tool arguments from state and persist the result into state."""

    def __init__(
        self,
        name: str,
        executor: ToolExecutor,
        arguments_builder: ArgumentsBuilder,
        result_key: str,
        requires_approval: bool = False,
    ) -> None:
        if not name.strip():
            raise ValueError("tool step name cannot be empty")
        if not result_key.strip():
            raise ValueError("result key cannot be empty")
        self.name = name
        self._executor = executor
        self._arguments_builder = arguments_builder
        self._result_key = result_key
        self._requires_approval = requires_approval

    def __call__(self, state: State) -> State:
        next_state = dict(state)
        arguments = self._arguments_builder(dict(state))
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be a dictionary")
        if self._requires_approval:
            decision = interrupt(
                {
                    "type": "tool_approval",
                    "tool": self.name,
                    "arguments": arguments,
                }
            )
            approved, arguments = self._resolve_approval(decision, arguments)
            if not approved:
                next_state[self._result_key] = {
                    "status": "rejected",
                    "reason": self._rejection_reason(decision),
                }
                return next_state
        result = self._executor.invoke(
            self.name,
            json.dumps(arguments, ensure_ascii=False),
        )
        next_state[self._result_key] = result
        return next_state

    @staticmethod
    def _resolve_approval(
        decision: Any,
        arguments: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        if decision is True or decision == "approve":
            return True, arguments
        if not isinstance(decision, dict):
            return False, arguments

        action = decision.get("action")
        if action == "approve":
            return True, arguments
        if action == "edit":
            edited = decision.get("arguments")
            if not isinstance(edited, dict):
                raise TypeError("edited tool arguments must be a dictionary")
            return True, edited
        return False, arguments

    @staticmethod
    def _rejection_reason(decision: Any) -> str:
        if isinstance(decision, dict):
            return str(decision.get("reason", "rejected by reviewer"))
        return "rejected by reviewer"


class AgentStep:
    """Read a question from state and persist an Agent result."""

    def __init__(
        self,
        agent: AgentRunner,
        question_key: str = "question",
        answer_key: str = "answer",
    ) -> None:
        self.name = "agent"
        self._agent = agent
        self._question_key = question_key
        self._answer_key = answer_key

    def __call__(self, state: State) -> State:
        question = state.get(self._question_key)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"state must contain non-empty {self._question_key}"
            )
        result = self._agent.run_result(question)
        next_state = dict(state)
        next_state[self._answer_key] = result.answer
        next_state["agent_result"] = {
            "rounds": result.rounds,
            "tool_calls": result.tool_calls,
            "stop_reason": str(result.stop_reason),
        }
        return next_state
