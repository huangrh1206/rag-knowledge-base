from __future__ import annotations
from typing import Any
from .state import TaskState
from .models import TaskStatus
from .planner import Planner
from .scheduler import SuperstepScheduler, SuperstepResult

def build_task_graph(planner: Planner, scheduler: SuperstepScheduler, *, checkpointer: Any = None, max_supersteps: int = 100):
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt
    builder = StateGraph(dict)
    def load_task(state): return state
    def plan_task(state):
        plan = planner.plan(state["goal"])
        return {**state, "plan": [s.__dict__ | {"status": s.status.value} for s in plan], "status": TaskStatus.RUNNING.value}
    def find_ready(state):
        current = TaskState.from_dict(state)
        return {**state, "ready_steps": scheduler.find_ready(current)}
    def human_approval(state):
        current = TaskState.from_dict(state)
        pending = next((s for s in current.plan if s.id in current.ready_steps and s.requires_approval), None)
        if pending is None:
            return {**state, "status": TaskStatus.RUNNING.value}
        decision = interrupt({"step_id": pending.id, "tool": pending.tool, "arguments": pending.arguments})
        approved = decision is True or (isinstance(decision, dict) and decision.get("action") in {"approve", "edit"})
        if not approved:
            return {**state, "status": TaskStatus.FAILED.value, "error": "human approval denied", "failed_steps": [pending.id]}
        if isinstance(decision, dict) and decision.get("action") == "edit":
            pending.arguments = dict(decision.get("arguments", pending.arguments))
        pending.requires_approval = False
        return {**state, "plan": [s.__dict__ | {"status": s.status.value} for s in current.plan], "status": TaskStatus.RUNNING.value}
    def execute(state):
        current = TaskState.from_dict(state); result = scheduler.run(current, current.ready_steps); return {**state, "_superstep": result.to_dict()}
    def merge(state):
        current = TaskState.from_dict(state); merged = scheduler.merge(current, SuperstepResult.from_dict(state["_superstep"])); return merged.to_dict()
    def route(state):
        if state.get("failed_steps"): return "fail"
        if not state.get("ready_steps"): return "end"
        return "execute_superstep" if state.get("superstep_id", 0) < max_supersteps else "fail"
    builder.add_node("load_task", load_task); builder.add_node("plan_task", plan_task); builder.add_node("find_ready_steps", find_ready)
    builder.add_node("execute_superstep", execute); builder.add_node("merge_results", merge); builder.add_node("human_approval", human_approval)
    builder.add_edge(START, "load_task"); builder.add_edge("load_task", "plan_task"); builder.add_edge("plan_task", "find_ready_steps")
    def approval_route(state):
        ready = set(state.get("ready_steps", []))
        return "human_approval" if any(s.get("id") in ready and s.get("requires_approval") for s in state.get("plan", [])) else "execute_superstep"
    builder.add_conditional_edges("find_ready_steps", approval_route, {"human_approval": "human_approval", "execute_superstep": "execute_superstep"})
    builder.add_edge("human_approval", "find_ready_steps"); builder.add_edge("execute_superstep", "merge_results")
    builder.add_conditional_edges("merge_results", route, {"execute_superstep": "find_ready_steps", "end": END, "fail": END})
    return builder.compile(checkpointer=checkpointer)
