from types import SimpleNamespace

from src.agent import KnowledgeAgent
from src.agent.executor import ToolExecutor
from src.agent.tools import ToolRegistry


class Retriever:
    def search(self, question: str) -> list:
        return []


class Gateway:
    def complete(self, messages, tools) -> SimpleNamespace:
        return SimpleNamespace(content="done", tool_calls=())


def test_agent_emits_model_events() -> None:
    events = []
    agent = KnowledgeAgent(
        None,
        "model",
        Retriever(),
        gateway=Gateway(),
        event_callback=lambda name, payload: events.append((name, payload)),
    )

    assert agent.run("question") == "done"
    assert [name for name, _ in events] == [
        "model_requested",
        "model_completed",
    ]


def test_executor_emits_tool_events() -> None:
    events = []

    class Echo:
        name = "echo"
        definition = {"type": "function", "function": {"name": "echo"}}

        def invoke(self, arguments: str) -> str:
            return "ok"

    executor = ToolExecutor(
        ToolRegistry([Echo()]),
        event_callback=lambda name, payload: events.append((name, payload)),
    )

    assert executor.invoke("echo", "{}", call_id="c1") == "ok"
    assert [name for name, _ in events] == [
        "tool_requested",
        "tool_completed",
    ]
