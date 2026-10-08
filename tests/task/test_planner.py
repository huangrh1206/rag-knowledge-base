import pytest
from src.task.models import TaskStep
from src.task.planner import StaticPlanner

def test_planner_rejects_cycles_and_unknown_tools():
    with pytest.raises(ValueError): StaticPlanner([TaskStep("a", "A", dependencies=["b"]), TaskStep("b", "B", dependencies=["a"])]).plan("x")
    with pytest.raises(ValueError): StaticPlanner([TaskStep("a", "A", tool="bad")], available_tools={"ok"}).plan("x")

