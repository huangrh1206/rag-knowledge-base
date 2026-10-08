from __future__ import annotations
from .models import TaskResult
from .models import TaskStep
from .runtime import TaskRuntime
from .tools import TaskToolRegistry, TaskToolExecutor
from .scheduler import SuperstepScheduler
from uuid import uuid4
from collections.abc import Callable
from typing import Any
from src.conversation import ConversationContext

class CalculatorTool:
    name, description = "calculator", "Evaluate arithmetic expressions"
    def invoke(self, arguments): return eval(str(arguments.get("expression", "0")), {"__builtins__": {}}, {})

class FileTool:
    name, description = "file", "Read or write a file"
    def invoke(self, arguments):
        from pathlib import Path
        path = Path(arguments["path"])
        if arguments.get("operation", "read") == "write": path.write_text(str(arguments.get("content", "")), encoding="utf-8"); return True
        return path.read_text(encoding="utf-8")

class HttpTool:
    name, description = "http", "HTTP adapter"
    def invoke(self, arguments): raise RuntimeError("HTTP tool requires an injected implementation")

class GeneralTaskAgent:
    def __init__(
        self,
        runtime: TaskRuntime | None = None,
        registry: TaskToolRegistry | None = None,
        responder: Callable[[str, list[dict[str, str]], TaskResult], str] | None = None,
        conversation: ConversationContext | None = None,
    ):
        self.registry = registry or TaskToolRegistry([CalculatorTool(), FileTool(), HttpTool()])
        self._uses_default_runtime = runtime is None
        self.runtime = runtime or TaskRuntime(scheduler=SuperstepScheduler(TaskToolExecutor(self.registry)))
        self.messages: list[dict[str, str]] = []
        self.responder = responder
        self.conversation = conversation

    def run(self, goal: str) -> TaskResult:
        plan = [TaskStep("goal", "goal", arguments={"value": goal})] if self._uses_default_runtime else None
        return self.runtime.start(uuid4().hex, goal, plan=plan)

    def chat(self, message: str) -> TaskResult:
        """Run one turn while retaining generic-agent conversation history."""
        if not message or not message.strip():
            raise ValueError("message cannot be empty")
        if self.conversation is not None:
            history = self.conversation.build_messages(message)
            self.messages = [
                {"role": str(item["role"]), "content": str(item.get("content", ""))}
                for item in history
                if item.get("role") != "system"
            ]
            responder_history = [dict(item) for item in history]
        else:
            self.messages.append({"role": "user", "content": message})
            responder_history = list(self.messages)
        context = "\n".join(f"{item['role']}: {item['content']}" for item in self.messages)
        result = self.run(context)
        answer = self.responder(message, responder_history, result) if self.responder else self._default_response(message, self.messages, result)
        result.final_answer = answer
        self.messages.append({"role": "assistant", "content": answer})
        if self.conversation is not None:
            self.conversation.record_turn(message, answer)
        return result

    @staticmethod
    def _default_response(message: str, messages: list[dict[str, str]], result: TaskResult) -> str:
        """Provide a readable local response when no model responder is injected."""
        normalized = message.strip().lower()
        if any(token in normalized for token in ("你是谁", "你是誰", "who are you")):
            return "我是通用长任务 Agent，负责拆解任务、调用通用工具，并支持多轮对话、暂停、恢复和人工审批。"
        if result.status.value == "completed":
            previous = [item["content"] for item in messages[:-1] if item["role"] == "user"]
            context = f"；前文：{' | '.join(previous[-2:])}" if previous else ""
            return f"已完成：{message.strip()}{context}"
        if result.error:
            return f"任务未完成：{result.error}"
        return f"任务状态：{result.status.value}"

    def reset_chat(self) -> None:
        self.messages.clear()
        if self.conversation is not None:
            self.conversation.reset()
