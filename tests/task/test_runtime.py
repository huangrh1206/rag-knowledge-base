from src.task.models import TaskStep, TaskStatus
from src.task.runtime import TaskRuntime
from src.task.evaluator import Evaluation
from src.task.planner import ModelPlanner

def test_runtime_runs_dependencies_and_restores_from_shared_checkpointer():
    from src.task.runtime import InMemorySaver
    saver = InMemorySaver()
    runtime = TaskRuntime(checkpointer=saver)
    result = runtime.start("t", "g", plan=[TaskStep("a", "A", arguments={"value": 1}), TaskStep("b", "B", dependencies=["a"], arguments={"value": 2})])
    assert result.status == TaskStatus.COMPLETED
    restored = TaskRuntime(checkpointer=saver)
    assert restored.get_state("t")["completed_steps"] == ["a", "b"]


def test_runtime_evaluates_and_replans_after_each_superstep():
    calls = []

    def model(request):
        calls.append(request)
        if not request["state"].get("completed_steps"):
            return [{"id": "a", "name": "first", "arguments": {"value": 1}}]
        return [
            {"id": "a", "name": "first", "arguments": {"value": 1}},
            {"id": "b", "name": "second", "dependencies": ["a"], "arguments": {"value": 2}},
        ]

    class NeedsSecondStep:
        def evaluate(self, goal, state):
            return Evaluation(sufficient="b" in state.completed_steps)

    runtime = TaskRuntime(
        planner=ModelPlanner(model),
        evaluator=NeedsSecondStep(),
    )
    result = runtime.start("replan", "do two things")

    assert result.status == TaskStatus.COMPLETED
    assert result.state["completed_steps"] == ["a", "b"]
    assert len(calls) == 2
    assert any(event["type"] == "evaluation" for event in result.state["events"])
    assert any(event["type"] == "replanned" for event in result.state["events"])
