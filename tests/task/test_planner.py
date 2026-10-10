import pytest
from src.task.models import TaskStep
from src.task.planner import ModelPlanner, StaticPlanner

def test_planner_rejects_cycles_and_unknown_tools():
    with pytest.raises(ValueError): StaticPlanner([TaskStep("a", "A", dependencies=["b"]), TaskStep("b", "B", dependencies=["a"])]).plan("x")
    with pytest.raises(ValueError): StaticPlanner([TaskStep("a", "A", tool="bad")], available_tools={"ok"}).plan("x")


def test_model_planner_receives_goal_state_tools_and_constraints():
    requests = []

    def model(request):
        requests.append(request)
        return {
            "steps": [
                {
                    "id": "weather",
                    "name": "Get weather",
                    "tool": "weather",
                    "arguments": {"city": "Nanjing"},
                }
            ]
        }

    planner = ModelPlanner(
        model,
        tool_definitions=[{"name": "weather", "description": "weather lookup"}],
        constraints={"max_parallel": 2},
    )
    steps = planner.plan("plan a trip", {"completed_steps": []})

    assert steps[0].tool == "weather"
    assert requests[0]["goal"] == "plan a trip"
    assert requests[0]["state"] == {"completed_steps": []}
    assert requests[0]["tools"][0]["name"] == "weather"
    assert requests[0]["constraints"]["max_parallel"] == 2


def test_model_planner_parses_json_and_validates_result():
    planner = ModelPlanner(
        lambda _: '{"steps": [{"id": "a", "name": "A"}]}',
        max_steps=1,
    )
    assert planner.plan("x")[0].id == "a"

    with pytest.raises(ValueError, match="unknown tool"):
        ModelPlanner(
            lambda _: [{"id": "a", "name": "A", "tool": "missing"}],
            available_tools={"known"},
        ).plan("x")

    with pytest.raises(ValueError, match="cyclic"):
        ModelPlanner(
            lambda _: {
                "steps": [
                    {"id": "a", "name": "A", "dependencies": ["b"]},
                    {"id": "b", "name": "B", "dependencies": ["a"]},
                ]
            }
        ).plan("x")


def test_model_planner_accepts_chat_gateway_content():
    class Response:
        content = '{"steps": [{"id": "a", "name": "A"}]}'

    class Gateway:
        def complete(self, messages, tools):
            assert messages[0]["role"] == "system"
            assert tools == []
            return Response()

    assert ModelPlanner(Gateway()).plan("x")[0].name == "A"
