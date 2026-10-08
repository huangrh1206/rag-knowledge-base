from src.task.graph import build_task_graph
from src.task.models import TaskStep
from src.task.planner import StaticPlanner
from src.task.scheduler import SuperstepScheduler
from src.task.state import TaskState

def test_graph_fallback_or_langgraph_graph_runs():
    graph = build_task_graph(StaticPlanner([TaskStep("a", "A", arguments={"value": 1})]), SuperstepScheduler())
    value = graph.invoke(TaskState("t", "g").to_dict())
    assert value["completed_steps"] == ["a"]

