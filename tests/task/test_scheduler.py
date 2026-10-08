from src.task.models import TaskStep
from src.task.scheduler import SuperstepScheduler
from src.task.state import TaskState

def test_independent_steps_share_a_superstep():
    state = TaskState("t", "g", plan=[TaskStep("a", "A", arguments={"value": 1}), TaskStep("b", "B", arguments={"value": 2})])
    scheduler = SuperstepScheduler()
    result = scheduler.run(state)
    scheduler.merge(state, result)
    assert {x.step_id for x in result.results} == {"a", "b"}
    assert state.completed_steps == ["a", "b"]

