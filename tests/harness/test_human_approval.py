import json

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agent.executor import ToolExecutor
from src.agent.tools import ToolRegistry
from src.harness import HarnessPolicy, HarnessRuntime, ToolStep


class EchoTool:
    name = "echo"
    definition = {"type": "function", "function": {"name": "echo"}}

    def invoke(self, arguments: str) -> str:
        return json.loads(arguments)["value"]


def test_tool_step_resumes_after_human_approval() -> None:
    runtime = HarnessRuntime(
        policy=HarnessPolicy(max_steps=2),
        checkpointer=InMemorySaver(),
    )
    executor = ToolExecutor(ToolRegistry([EchoTool()]))
    step = ToolStep(
        "echo",
        executor,
        lambda state: {"value": "before-review"},
        "tool_result",
        requires_approval=True,
    )

    paused = runtime.run("approval-1", [step])
    resumed = runtime.run(
        "approval-1",
        [step],
        resume=True,
        resume_value={"action": "edit", "arguments": {"value": "approved"}},
    )

    assert paused.status == "interrupted"
    assert resumed.status == "completed"
    assert resumed.state["tool_result"] == "approved"
