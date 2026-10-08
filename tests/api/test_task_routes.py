import pytest

def test_task_routes_register_when_fastapi_available():
    fastapi = pytest.importorskip("fastapi")
    from src.api.task_routes import register_task_routes
    from src.task.runtime import TaskRuntime
    app = fastapi.FastAPI()
    register_task_routes(app, TaskRuntime())
    paths = {route.path for route in app.routes}
    assert "/tasks" in paths and "/tasks/{task_id}/resume" in paths
