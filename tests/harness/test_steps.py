import json

from src.agent.tools import ToolRegistry
from src.harness import AgentStep, FunctionStep, HarnessRuntime, ToolStep
from src.agent.executor import ToolExecutor
from src.agent.types import AgentResult, AgentStopReason


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
    runtime = HarnessRuntime()
    executor = ToolExecutor(
        ToolRegistry([EchoTool()]),
        event_callback=runtime.recorder("tool-run"),
    )
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

    result = runtime.run("tool-run", steps)

    assert result.status == "completed"
    assert result.state["tool_result"] == "from-state"
    event_types = [event.event_type for event in result.events]
    assert "tool_requested" in event_types
    assert "tool_completed" in event_types


def test_agent_step_maps_question_and_result() -> None:
    class FakeAgent:
        def run_result(self, question: str) -> AgentResult:
            return AgentResult(
                answer=f"answer for {question}",
                rounds=2,
                tool_calls=1,
                stop_reason=AgentStopReason.COMPLETED,
            )

    step = AgentStep(FakeAgent())
    state = step({"question": "What is MCP?"})

    assert state["answer"] == "answer for What is MCP?"
    assert state["agent_result"]["tool_calls"] == 1
