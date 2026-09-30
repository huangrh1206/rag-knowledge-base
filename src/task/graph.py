from __future__ import annotations
from typing import Any
from .state import TaskState
from .models import TaskStatus
from .planner import Planner
from .scheduler import SuperstepScheduler

def build_task_graph(planner: Planner, scheduler: SuperstepScheduler, *, checkpointer: Any = None, max_supersteps: int = 100):
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError:
        return _FallbackGraph(planner, scheduler, max_supersteps)
    builder = StateGraph(dict)
    def load_task(state): return state
    def plan_task(state):
        plan = planner.plan(state["goal"])
        return {"plan": [s.__dict__ for s in plan], "status": TaskStatus.RUNNING.value}
    def find_ready(state):
        current = TaskState.from_dict(state)
        return {"ready_steps": scheduler.find_ready(current)}
    def execute(state):
        current = TaskState.from_dict(state); result = scheduler.run(current, current.ready_steps); return {"_superstep": result}
    def merge(state):
        current = TaskState.from_dict(state); merged = scheduler.merge(current, state["_superstep"]); return merged.to_dict()
    def route(state):
        if state.get("failed_steps"): return "fail"
        if not state.get("ready_steps"): return "end"
        return "execute_superstep" if state.get("superstep_id", 0) < max_supersteps else "fail"
    builder.add_node("load_task", load_task); builder.add_node("plan_task", plan_task); builder.add_node("find_ready_steps", find_ready)
    builder.add_node("execute_superstep", execute); builder.add_node("merge_results", merge)
    builder.add_edge(START, "load_task"); builder.add_edge("load_task", "plan_task"); builder.add_edge("plan_task", "find_ready_steps")
    builder.add_edge("find_ready_steps", "execute_superstep"); builder.add_edge("execute_superstep", "merge_results")
    builder.add_conditional_edges("merge_results", route, {"execute_superstep": "find_ready_steps", "end": END, "fail": END})
    return builder.compile(checkpointer=checkpointer)

class _FallbackGraph:
    def __init__(self, planner, scheduler, max_supersteps): self.planner, self.scheduler, self.max_supersteps = planner, scheduler, max_supersteps
    def invoke(self, value, config=None):
        state = value if isinstance(value, TaskState) else TaskState.from_dict(value)
        state.plan = self.planner.plan(state.goal); state.status = TaskStatus.RUNNING
        while state.superstep_id < self.max_supersteps:
            state.ready_steps = self.scheduler.find_ready(state)
            if not state.ready_steps: state.status = TaskStatus.FAILED if state.failed_steps else TaskStatus.COMPLETED; return state.to_dict()
            self.scheduler.merge(state, self.scheduler.run(state, state.ready_steps))
            if state.failed_steps: state.status = TaskStatus.FAILED; return state.to_dict()
        state.status = TaskStatus.FAILED; state.error = "maximum supersteps exceeded"; return state.to_dict()
