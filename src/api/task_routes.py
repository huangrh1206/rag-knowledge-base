from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field
from src.task.runtime import TaskRuntime

class TaskRequest(BaseModel): goal: str = Field(min_length=1)
class ResumeRequest(BaseModel): value: Any = None

def register_task_routes(app: Any, runtime: TaskRuntime) -> Any:
    @app.post("/tasks")
    def create(request: TaskRequest): return runtime.start(request.goal.__hash__().__str__(), request.goal).state
    @app.get("/tasks/{task_id}")
    def get(task_id: str): return runtime.get_state(task_id)
    @app.post("/tasks/{task_id}/resume")
    def resume(task_id: str, request: ResumeRequest | None = None): return runtime.resume(task_id, request.value if request else None).state
    @app.post("/tasks/{task_id}/pause")
    def pause(task_id: str): return runtime.pause(task_id).state
    @app.post("/tasks/{task_id}/cancel")
    def cancel(task_id: str): return runtime.cancel(task_id).state
    @app.get("/tasks/{task_id}/events")
    def events(task_id: str): return runtime.get_state(task_id).get("events", [])
    return app

