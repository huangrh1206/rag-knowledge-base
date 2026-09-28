"""Conversation context that combines short- and long-term memory."""

from typing import Any

from src.memory.long_term import LongTermMemory, MemoryExtractor
from src.memory.models import MemoryScope
from src.memory.short_term import ShortTermMemory


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

    def build_messages(self, question: str) -> list[dict[str, Any]]:
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
