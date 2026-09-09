"""Reusable steps for connecting Harness state to Agent tools."""

import json
from collections.abc import Callable
from typing import Any

from src.agent.executor import ToolExecutor

State = dict[str, Any]
ArgumentsBuilder = Callable[[State], dict[str, Any]]
StepHandler = Callable[[State], State]


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
    ) -> None:
        if not name.strip():
            raise ValueError("tool step name cannot be empty")
        if not result_key.strip():
            raise ValueError("result key cannot be empty")
        self.name = name
        self._executor = executor
        self._arguments_builder = arguments_builder
        self._result_key = result_key

    def __call__(self, state: State) -> State:
        next_state = dict(state)
        arguments = self._arguments_builder(dict(state))
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be a dictionary")
        result = self._executor.invoke(
            self.name,
            json.dumps(arguments, ensure_ascii=False),
        )
        next_state[self._result_key] = result
        return next_state
