from types import SimpleNamespace

import pytest

from src.task.chat import TaskChatResponder


class FakeCompletions:
    def __init__(self, content: str | None = "模型回答"):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content, tool_calls=[]))]
        )


def test_task_chat_responder_uses_history_without_tools():
    api = FakeCompletions()
    responder = TaskChatResponder(api, "chat-model")
    answer = responder("继续", [{"role": "user", "content": "前文"}], None)
    assert answer == "模型回答"
    assert api.calls[0]["tools"] == []
    assert api.calls[0]["messages"][1]["content"] == "前文"


def test_task_chat_responder_surfaces_model_failure():
    api = FakeCompletions(None)
    with pytest.raises(RuntimeError, match="空答案"):
        TaskChatResponder(api, "chat-model")("问题", [], None)
