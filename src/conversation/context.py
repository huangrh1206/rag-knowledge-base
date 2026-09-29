"""Conversation context that combines short- and long-term memory."""

from enum import StrEnum
from typing import Any

from src.memory.long_term import LongTermMemory, MemoryExtractor
from src.memory.models import MemoryScope
from src.memory.short_term import ShortTermMemory


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class ConversationContext:
    def __init__(
        self,
        scope: MemoryScope,
        *,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        self.scope = scope
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term
        self.extractor = extractor
        self.status = ConversationStatus.ACTIVE
        self.turn_count = 0

    def build_messages(self, question: str) -> list[dict[str, Any]]:
        if self.status is not ConversationStatus.ACTIVE:
            raise RuntimeError(
                f"conversation is not active: {self.status.value}"
            )
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        messages = self.short_term.messages()
        if not messages or messages[-1].get("role") != "user":
            if self.long_term is not None:
                memories = self.long_term.recall(self.scope, question)
                if memories:
                    memory_text = "\n".join(f"- {item.content}" for item in memories)
                    messages.insert(
                        0,
                        {"role": "system", "content": f"Relevant memory:\n{memory_text}"},
                    )
            messages.append({"role": "user", "content": question})
        return messages

    def record(self, messages: list[dict[str, Any]]) -> None:
        self.short_term.clear()
        self.short_term.extend(messages)
        if self.long_term is not None and self.extractor is not None:
            self.long_term.extract_and_remember(
                self.scope,
                messages,
                self.extractor,
            )
        self.turn_count += 1

    def record_turn(self, question: str, answer: str) -> None:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self.short_term.extend(messages)
        if self.long_term is not None and self.extractor is not None:
            self.long_term.extract_and_remember(
                self.scope,
                messages,
                self.extractor,
            )
        self.turn_count += 1

    def pause(self) -> None:
        if self.status is ConversationStatus.COMPLETED:
            raise RuntimeError("completed conversation cannot be paused")
        self.status = ConversationStatus.PAUSED

    def resume(self) -> None:
        if self.status is ConversationStatus.COMPLETED:
            raise RuntimeError("completed conversation cannot be resumed")
        self.status = ConversationStatus.ACTIVE

    def complete(self) -> None:
        self.status = ConversationStatus.COMPLETED

    def reset(self) -> None:
        self.short_term.clear()
        self.status = ConversationStatus.ACTIVE
        self.turn_count = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "turn_count": self.turn_count,
            "messages": self.short_term.messages(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.short_term.clear()
        self.short_term.extend(snapshot.get("messages", []))
        self.status = ConversationStatus(snapshot.get("status", "active"))
        self.turn_count = int(snapshot.get("turn_count", 0))
