import json

import pytest

from src.agent.executor import ToolExecutionPolicy, ToolExecutor
from src.agent.tools import ToolRegistry


class FakeTool:
    name = "fake"
    definition = {
        "type": "function",
        "function": {"name": "fake"},
    }

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or ["ok"])
        self.error = error
        self.calls = 0

    def invoke(self, arguments: str) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def test_executor_deduplicates_completed_call_id() -> None:
    tool = FakeTool()
    executor = ToolExecutor(ToolRegistry([tool]))

    assert executor.invoke("fake", "{}", call_id="call-1") == "ok"
    assert executor.invoke("fake", "{}", call_id="call-1") == "ok"
    assert tool.calls == 1


def test_executor_retries_only_idempotent_tools() -> None:
    tool = FakeTool(error=RuntimeError("temporary"))
    registry = ToolRegistry([tool])
    executor = ToolExecutor(
        registry,
        policies={
            "fake": ToolExecutionPolicy(
                max_retries=2,
                idempotent=True,
            )
        },
    )

    result = json.loads(executor.invoke("fake", "{}"))

    assert result["error"] == "temporary"
    assert tool.calls == 3


def test_policy_rejects_retry_for_non_idempotent_tool() -> None:
    with pytest.raises(ValueError, match="idempotent"):
        ToolExecutionPolicy(max_retries=1)


def test_executor_requires_approval_for_protected_tool() -> None:
    tool = FakeTool()
    executor = ToolExecutor(
        ToolRegistry([tool]),
        policies={
            "fake": ToolExecutionPolicy(requires_approval=True)
        },
        approval=lambda name, arguments: False,
    )

    result = json.loads(executor.invoke("fake", "{}"))

    assert result["error"] == "tool approval denied"
    assert tool.calls == 0


def test_executor_returns_unknown_tool_error() -> None:
    result = json.loads(
        ToolExecutor(ToolRegistry()).invoke("missing", "{}")
    )

    assert result["error"] == "unknown tool"
