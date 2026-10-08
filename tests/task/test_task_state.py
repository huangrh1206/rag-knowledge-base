import json
import pytest
from src.task.models import StepStatus, TaskStep
from src.task.state import TaskState, merge_state

def test_state_serializes_and_preserves_fields():
    state = TaskState("t", "goal", plan=[TaskStep("a", "A")], artifacts={"x": 1})
    restored = TaskState.from_dict(json.loads(json.dumps(state.to_dict())))
    merge_state(restored, {"error": "e"})
    assert restored.artifacts == {"x": 1} and restored.plan[0].status == StepStatus.PENDING

def test_step_transitions_are_checked():
    step = TaskStep("a", "A")
    step.transition(StepStatus.READY); step.transition(StepStatus.RUNNING); step.transition(StepStatus.COMPLETED)
    with pytest.raises(ValueError): step.transition(StepStatus.RUNNING)

