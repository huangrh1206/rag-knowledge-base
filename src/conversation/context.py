"""Conversation context that combines short- and long-term memory."""

from enum import StrEnum
from typing import Any

from src.memory.long_term import (
    LongTermMemory,
    MemoryExtractor,
    RuleBasedMemoryExtractor,
)
from src.memory.models import MemoryScope
from src.memory.short_term import ShortTermMemory, default_summarizer
from src.conversation.store import ConversationStore, SQLiteConversationStore


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class ConversationContext:
    @classmethod
    def from_settings(
        cls,
        scope: MemoryScope,
        *,
        settings=None,
        extractor: MemoryExtractor | None = None,
        session_id: str = "default",
    ) -> "ConversationContext":
        from src.config import Settings
        from src.memory.factory import create_memory_store
        from src.memory.long_term import MemoryWritePolicy

        settings = settings or Settings.from_env()
        return cls(
            scope,
            short_term=ShortTermMemory(
                max_messages=settings.memory_max_messages,
                max_chars=settings.memory_max_chars,
                max_tokens=settings.memory_max_tokens,
                summarizer=default_summarizer if settings.memory_summarization_enabled else None,
            ),
            long_term=LongTermMemory(
                create_memory_store(settings),
                policy=MemoryWritePolicy(
                    min_confidence=settings.memory_min_confidence,
                ),
            ),
            # The production conversation path must extract memories by default.
            # Callers can still inject a model-backed or test extractor explicitly.
            extractor=extractor or RuleBasedMemoryExtractor(),
            session_id=session_id,
            persistence=SQLiteConversationStore(settings.memory_sqlite_path),
        )

    def __init__(
        self,
        scope: MemoryScope,
        *,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        extractor: MemoryExtractor | None = None,
        session_id: str | None = None,
        persistence: ConversationStore | None = None,
    ) -> None:
        self.scope = scope
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term
        self.extractor = extractor
        if session_id is not None and not session_id.strip():
            raise ValueError("session_id must be non-empty")
        self.session_id = session_id
        self.persistence = persistence
        self.status = ConversationStatus.ACTIVE
        self.turn_count = 0
        if self.session_id is not None and self.persistence is not None:
            snapshot = self.persistence.load(self.scope, self.session_id)
            if snapshot is not None:
                self.restore(snapshot)
                self._extract_long_term(snapshot.get("messages", []))

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
        self._extract_long_term(messages)
        self.turn_count += 1
        self._persist()

    def record_turn(self, question: str, answer: str) -> None:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self.short_term.extend(messages)
        self._extract_long_term(messages)
        self.turn_count += 1
        self._persist()

    def pause(self) -> None:
        if self.status is ConversationStatus.COMPLETED:
            raise RuntimeError("completed conversation cannot be paused")
        self.status = ConversationStatus.PAUSED
        self._persist()

    def resume(self) -> None:
        if self.status is ConversationStatus.COMPLETED:
            raise RuntimeError("completed conversation cannot be resumed")
        self.status = ConversationStatus.ACTIVE
        self._persist()

    def complete(self) -> None:
        self.status = ConversationStatus.COMPLETED
        self._persist()

    def reset(self) -> None:
        self.short_term.clear()
        self.status = ConversationStatus.ACTIVE
        self.turn_count = 0
        self._persist()

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

    def _persist(self) -> None:
        if self.session_id is not None and self.persistence is not None:
            self.persistence.save(self.scope, self.session_id, self.snapshot())

    def _extract_long_term(self, messages: list[dict[str, Any]]) -> None:
        if self.long_term is not None and self.extractor is not None:
            self.long_term.extract_and_remember(
                self.scope,
                messages,
                self.extractor,
            )
