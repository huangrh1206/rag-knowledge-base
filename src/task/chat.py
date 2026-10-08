"""Model-backed multi-turn chat for the general task agent.

This module deliberately knows nothing about retrieval or RAG tools.
"""
from __future__ import annotations

from typing import Any

from src.agent.model_gateway import ChatCompletionsGateway
from src.agent.types import AgentModelError


class TaskChatResponder:
    def __init__(self, api: Any, model: str, *, system_prompt: str | None = None) -> None:
        self.gateway = ChatCompletionsGateway(api, model)
        self.system_prompt = system_prompt or (
            "你是通用长任务 Agent。回答普通问题时直接、清晰地回答；"
            "规划任务时先澄清缺失信息，不要声称已经执行未执行的操作。"
        )

    def __call__(self, message: str, history: list[dict[str, str]], _result: Any) -> str:
        messages: list[dict[str, object]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        try:
            response = self.gateway.complete(messages, ())
        except AgentModelError as exc:
            raise RuntimeError("通用 Agent 模型调用失败，请检查 API 配置和网络连接") from exc
        if not response.content or not response.content.strip():
            raise RuntimeError("通用 Agent 模型返回了空答案")
        return response.content.strip()
