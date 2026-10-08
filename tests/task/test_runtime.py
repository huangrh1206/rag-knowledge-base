from src.task.models import TaskStep, TaskStatus
from src.task.runtime import TaskRuntime

def test_runtime_runs_dependencies_and_restores_from_shared_checkpointer():
    from src.task.runtime import InMemorySaver
    saver = InMemorySaver()
    runtime = TaskRuntime(checkpointer=saver)
    result = runtime.start("t", "g", plan=[TaskStep("a", "A", arguments={"value": 1}), TaskStep("b", "B", dependencies=["a"], arguments={"value": 2})])
    assert result.status == TaskStatus.COMPLETED
    restored = TaskRuntime(checkpointer=saver)
    assert restored.get_state("t")["completed_steps"] == ["a", "b"]

