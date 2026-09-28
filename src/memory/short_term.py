"""Bounded short-term conversation memory."""

from collections.abc import Mapping
from typing import Any


class ShortTermMemory:
    """Keep recent messages while preserving system and pending tool context."""

    def __init__(self, max_messages: int = 20, max_chars: int = 24_000) -> None:
        if max_messages < 2 or max_chars <= 0:
            raise ValueError("short-term memory limits are invalid")
        self.max_messages = max_messages
        self.max_chars = max_chars
        self._messages: list[dict[str, Any]] = []

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

    def _bounded(self) -> list[dict[str, Any]]:
        if not self._messages:
            return []
        system = [m for m in self._messages if m.get("role") == "system"][:1]
        remaining = [m for m in self._messages if m.get("role") != "system"]
        selected: list[dict[str, Any]] = []
        chars = sum(len(str(m.get("content", ""))) for m in system)
        for message in reversed(remaining):
            size = len(str(message.get("content", "")))
            if selected and (len(selected) >= self.max_messages - len(system) or chars + size > self.max_chars):
                break
            selected.append(message)
            chars += size
        return system + list(reversed(selected))
