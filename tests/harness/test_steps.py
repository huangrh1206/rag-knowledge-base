import json

from src.agent.tools import ToolRegistry
from src.harness import FunctionStep, HarnessRuntime, ToolStep
from src.agent.executor import ToolExecutor


class EchoTool:
    name = "echo"
    definition = {
        "type": "function",
        "function": {"name": "echo"},
    }

    def invoke(self, arguments: str) -> str:
        return json.loads(arguments)["value"]


def test_function_step_transforms_state() -> None:
    step = FunctionStep(
        "set_question",
        lambda state: {**state, "question": "hello"},
    )

    assert step({}) == {"question": "hello"}


def test_tool_step_connects_state_to_local_tool() -> None:
    executor = ToolExecutor(ToolRegistry([EchoTool()]))
    steps = [
        FunctionStep(
            "prepare",
            lambda state: {**state, "value": "from-state"},
        ),
        ToolStep(
            "echo",
            executor,
            lambda state: {"value": state["value"]},
            "tool_result",
        ),
    ]

    result = HarnessRuntime().run("tool-run", steps)

    assert result.status == "completed"
    assert result.state["tool_result"] == "from-state"
