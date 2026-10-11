"""Model-backed multi-turn chat for the general task agent.

This module deliberately knows nothing about retrieval or RAG tools.
"""
from __future__ import annotations

import json
from typing import Any

from src.agent.model_gateway import ChatCompletionsGateway
from src.agent.types import AgentModelError


class TaskChatResponder:
    def __init__(self, api: Any, model: str, *, system_prompt: str | None = None, tool_definitions: list[dict[str, Any]] | None = None, executor: Any | None = None, max_tool_rounds: int = 8, event_callback: Any | None = None) -> None:
        self.gateway = ChatCompletionsGateway(api, model)
        self.tool_definitions = tool_definitions or []
        self.executor = executor
        self.max_tool_rounds = max_tool_rounds
        self.event_callback = event_callback
        if system_prompt is None:
            tool_summary = json.dumps(
                [{"name": item.get("name"), "description": item.get("description", "")} for item in self.tool_definitions],
                ensure_ascii=False,
            ) if self.tool_definitions else "（当前没有已注册工具）"
            system_prompt = (
                "你是通用长任务 Agent。回答普通问题时直接、清晰地回答；"
                "规划任务时先澄清缺失信息，不要声称执行了尚未执行的操作。"
                f"当前已注册的工具如下：{tool_summary}。"
                "如果用户询问可用工具，必须如实列出这些工具；不要声称没有外部工具。"
            )
        self.system_prompt = system_prompt or (
            "你是通用长任务 Agent。回答普通问题时直接、清晰地回答；"
            "规划任务时先澄清缺失信息，不要声称已经执行未执行的操作。"
        )

    def available_tools_answer(self) -> str:
        if not self.tool_definitions:
            return "当前没有已注册的外部工具。"
        lines = ["当前可以调用的工具："]
        for item in self.tool_definitions:
            lines.append(f"- {item.get('name')}: {item.get('description', '')}")
        return "\n".join(lines)

    def __call__(self, message: str, history: list[dict[str, str]], _result: Any) -> str:
        normalized = message.strip().lower()
        if any(token in normalized for token in ("可以调用哪些工具", "有哪些工具", "可用工具", "工具列表", "what tools")):
            return self.available_tools_answer()
        messages: list[dict[str, object]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        tools = self._openai_tools()
        for _ in range(self.max_tool_rounds + 1):
            round_id = _ + 1
            self._emit("model_requested", {"round": round_id, "tool_count": len(tools)})
            try:
                response = self.gateway.complete(messages, tools)
            except AgentModelError as exc:
                raise RuntimeError("通用 Agent 模型调用失败，请检查 API 配置和网络连接") from exc
            if not response.tool_calls:
                if not response.content or not response.content.strip():
                    raise RuntimeError("通用 Agent 模型返回了空答案")
                self._emit("final_answer", {"content": response.content.strip(), "round": round_id})
                return response.content.strip()
            if self.executor is None:
                raise RuntimeError("模型请求调用工具，但聊天执行器未配置")
            assistant_tool_calls = []
            for call in response.tool_calls:
                arguments = self._parse_arguments(call.arguments)
                assistant_tool_calls.append({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                })
            messages.append({"role": "assistant", "content": response.content or "", "tool_calls": assistant_tool_calls})
            for call in response.tool_calls:
                arguments = self._parse_arguments(call.arguments)
                self._emit("tool_requested", {"tool": call.name, "arguments": arguments, "call_id": call.id, "round": round_id})
                try:
                    self._emit("tool_started", {"tool": call.name, "call_id": call.id, "round": round_id})
                    value = self.executor.invoke(call.name, arguments, idempotency_key=f"chat:{call.id}")
                    content = json.dumps(value, ensure_ascii=False, default=str)
                    self._emit("tool_completed", {"tool": call.name, "call_id": call.id, "result": value, "round": round_id})
                except Exception as exc:
                    content = json.dumps({"error": str(exc), "tool": call.name}, ensure_ascii=False)
                    self._emit("tool_failed", {"tool": call.name, "call_id": call.id, "error": str(exc), "round": round_id})
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})
        raise RuntimeError("tool call rounds exceeded")

    def _emit(self, name: str, payload: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(name, payload)

    def _openai_tools(self) -> list[dict[str, Any]]:
        result = []
        for item in self.tool_definitions:
            name = item.get("name")
            if not name:
                continue
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": item.get("description", ""),
                    "parameters": item.get("parameters") or {"type": "object", "additionalProperties": True},
                },
            })
        return result

    @staticmethod
    def _parse_arguments(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not value:
            return {}
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed
