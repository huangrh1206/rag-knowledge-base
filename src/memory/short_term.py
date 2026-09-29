"""Bounded short-term conversation memory."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol


class TokenEstimator(Protocol):
    def estimate(self, message: Mapping[str, Any]) -> int: ...


class ApproximateTokenEstimator:
    """Dependency-free estimate; callers may inject a model tokenizer."""

    def estimate(self, message: Mapping[str, Any]) -> int:
        content = str(message.get("content", ""))
        tool_calls = message.get("tool_calls", ())
        return max(1, (len(content) + 3) // 4) + len(tool_calls) * 8


class ShortTermMemory:
    """Keep recent messages while preserving system and pending tool context."""

    def __init__(
        self,
        max_messages: int = 20,
        max_chars: int = 24_000,
        *,
        max_tokens: int | None = None,
        token_estimator: TokenEstimator | None = None,
        summarizer: Callable[[list[dict[str, Any]]], str] | None = None,
    ) -> None:
        if max_messages < 2 or max_chars <= 0 or (
            max_tokens is not None and max_tokens <= 0
        ):
            raise ValueError("short-term memory limits are invalid")
        self.max_messages = max_messages
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.token_estimator = token_estimator or ApproximateTokenEstimator()
        self.summarizer = summarizer
        self._messages: list[dict[str, Any]] = []
        self._summary = ""

    def append(self, message: Mapping[str, Any]) -> None:
        if "role" not in message:
            raise ValueError("message role is required")
        self._messages.append(dict(message))

    def extend(self, messages: list[Mapping[str, Any]]) -> None:
        for message in messages:
            self.append(message)

    def messages(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self._bounded()]

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""

    def _bounded(self) -> list[dict[str, Any]]:
        if not self._messages:
            return []
        system = [m for m in self._messages if m.get("role") == "system"][:1]
        if self._summary:
            system.append(
                {"role": "system", "content": f"Conversation summary:\n{self._summary}"}
            )
        remaining = [m for m in self._messages if m.get("role") != "system"]
        units = self._message_units(remaining)
        selected: list[dict[str, Any]] = []
        chars = sum(len(str(m.get("content", ""))) for m in system)
        tokens = sum(self.token_estimator.estimate(m) for m in system)
        dropped: list[dict[str, Any]] = []
        for unit in reversed(units):
            size = sum(len(str(m.get("content", ""))) for m in unit)
            unit_tokens = sum(self.token_estimator.estimate(m) for m in unit)
            too_many_messages = len(selected) + len(unit) > self.max_messages - len(system)
            too_many_chars = chars + size > self.max_chars
            too_many_tokens = self.max_tokens is not None and tokens + unit_tokens > self.max_tokens
            if selected and (too_many_messages or too_many_chars or too_many_tokens):
                dropped[0:0] = unit
                break
            selected[0:0] = unit
            chars += size
            tokens += unit_tokens
        if dropped and self.summarizer is not None:
            summary = self.summarizer(dropped).strip()
            if summary:
                self._summary = summary
                system.append(
                    {"role": "system", "content": f"Conversation summary:\n{summary}"}
                )
        return system + selected

    @staticmethod
    def _message_units(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        units: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            unit = [message]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                expected_ids = {
                    call["id"]
                    for call in message["tool_calls"]
                    if isinstance(call, Mapping) and "id" in call
                }
                index += 1
                while (
                    index < len(messages)
                    and messages[index].get("role") == "tool"
                    and messages[index].get("tool_call_id") in expected_ids
                ):
                    unit.append(messages[index])
                    index += 1
                units.append(unit)
                continue
            units.append(unit)
            index += 1
        return units
