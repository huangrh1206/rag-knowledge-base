"""Long-term memory facade and pluggable extraction policies."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from src.memory.models import MemoryItem, MemoryScope
from src.memory.store import MemoryStore


@dataclass(frozen=True)
class ExtractedMemory:
    content: str
    memory_type: str = "fact"
    key: str | None = None
    importance: float = 0.5
    metadata: dict[str, Any] | None = None


class MemoryExtractor(Protocol):
    def extract(self, messages: Iterable[dict[str, Any]]) -> list[ExtractedMemory]: 
        ...


class RuleBasedMemoryExtractor:
    """Conservative baseline extractor for explicit user preferences/facts."""

    _patterns = (
        ("prefers", "preference"),
        ("prefer", "preference"),
        ("\u559c\u6b22", "preference"),
        ("\u504f\u597d", "preference"),
        ("\u6211\u7684\u540d\u5b57\u662f", "identity"),
        ("my name is", "identity"),
    )

    def extract(self, messages: Iterable[dict[str, Any]]) -> list[ExtractedMemory]:
        memories: list[ExtractedMemory] = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content", "")).strip()
            lowered = content.lower()
            for marker, memory_type in self._patterns:
                if marker.lower() not in lowered:
                    continue
                key = "identity" if memory_type == "identity" else "preference"
                memories.append(
                    ExtractedMemory(
                        content=content,
                        memory_type=memory_type,
                        key=key,
                        importance=0.8,
                        metadata={"source": "rule_based"},
                    )
                )
                break
        return memories


class LongTermMemory:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def remember(
        self,
        scope: MemoryScope,
        content: str,
        *,
        memory_type: str = "fact",
        key: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, object] | None = None,
    ) -> MemoryItem:
        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            scope=scope,
            key=key,
            importance=importance,
            metadata=metadata or {},
        )
        return self.store.upsert(item)

    def recall(
        self,
        scope: MemoryScope,
        query: str = "",
        *,
        memory_type: str | None = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        return self.store.search(scope, query, memory_type=memory_type, limit=limit)

    def forget(self, scope: MemoryScope, memory_id: str) -> None:
        self.store.delete(scope, memory_id)

    def extract_and_remember(
        self,
        scope: MemoryScope,
        messages: Iterable[dict[str, Any]],
        extractor: MemoryExtractor,
    ) -> list[MemoryItem]:
        return [
            self.remember(
                scope,
                extracted.content,
                memory_type=extracted.memory_type,
                key=extracted.key,
                importance=extracted.importance,
                metadata=extracted.metadata,
            )
            for extracted in extractor.extract(messages)
        ]
