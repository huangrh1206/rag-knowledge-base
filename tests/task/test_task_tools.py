import pytest
from src.task.tools import TaskToolExecutor, TaskToolRegistry, ToolExecutionError

class Echo:
    name, description = "echo", "echo"
    def invoke(self, arguments): return arguments["value"]

def test_tool_execution_and_idempotency():
    executor = TaskToolExecutor(TaskToolRegistry([Echo()]))
    assert executor.invoke("echo", {"value": 2}, idempotency_key="x") == 2
    assert executor.invoke("echo", {"value": 3}, idempotency_key="x") == 2
    assert executor.audit_log

def test_unknown_tool_is_wrapped():
    with pytest.raises(ToolExecutionError): TaskToolExecutor(TaskToolRegistry()).invoke("missing", {})

