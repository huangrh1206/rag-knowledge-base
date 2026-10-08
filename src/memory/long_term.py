"""Long-term memory facade and conservative extraction policies."""

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Protocol

from src.memory.models import MemoryItem, MemoryScope
from src.memory.store import MemoryStore


@dataclass(frozen=True)
class ExtractedMemory:
    content: str
    memory_type: str = "fact"
    key: str | None = None
    importance: float = 0.5
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None


class MemoryExtractor(Protocol):
    def extract(self, messages: Iterable[dict[str, Any]]) -> list[ExtractedMemory]: ...


class RuleBasedMemoryExtractor:
    """Extract stable, user-authored facts from ordinary statements."""

    _preference_markers = ("prefer", "喜欢", "偏好", "习惯", "preference")
    _identity_markers = ("我的名字是", "我叫", "my name is")
    _remember_markers = ("记住", "请记住", "牢记", "remember")
    _fact_markers = {
        "family": ("妻子", "丈夫", "老公", "老婆", "父亲", "母亲", "爸爸", "妈妈", "儿子", "女儿", "家人"),
        "school": ("学校", "大学", "学院", "就读", "上学", "毕业于"),
        "occupation": ("工作", "职业", "任职", "就职", "是一名", "从事"),
        "location": ("住在", "居住", "所在地", "位于", "来自", "家在"),
    }

    def extract(self, messages: Iterable[dict[str, Any]]) -> list[ExtractedMemory]:
        memories: list[ExtractedMemory] = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            lowered = content.casefold()
            explicit = any(marker in lowered for marker in self._remember_markers)
            segments = [
                part.strip(" \t，。！？；：、,.!?;:\n")
                for part in re.split(r"[。！？；：\n]+", content)
                if part.strip(" \t，。！？；：、,.!?;:\n")
            ]
            for segment in segments:
                segment_lower = segment.casefold()
                if any(marker in segment_lower for marker in self._identity_markers):
                    memory_type, key = "identity", "identity"
                elif any(marker in segment_lower for marker in self._preference_markers):
                    memory_type, key = "preference", "preference"
                else:
                    memory_type, key = self._classify_fact(segment_lower)
                    if memory_type is None and not explicit:
                        continue
                    if memory_type is None:
                        digest = hashlib.sha1(segment.encode("utf-8")).hexdigest()[:12]
                        memory_type, key = "fact", f"fact:{digest}"
                memories.append(ExtractedMemory(segment, memory_type, key, 0.8, 1.0, {"source": "rule_based", "explicit": explicit}))
        return memories

    @classmethod
    def _classify_fact(cls, content: str) -> tuple[str | None, str | None]:
        for memory_type, markers in cls._fact_markers.items():
            if any(marker in content for marker in markers):
                return memory_type, memory_type
        return None, None


class LongTermMemory:
    def __init__(self, store: MemoryStore, *, policy: "MemoryWritePolicy | None" = None) -> None:
        self.store = store
        self.policy = policy or MemoryWritePolicy()

    def remember(self, scope: MemoryScope, content: str, *, memory_type: str = "fact", key: str | None = None, importance: float = 0.5, metadata: dict[str, object] | None = None, expires_at: str | None = None) -> MemoryItem:
        self.policy.validate(content, importance=importance, confidence=1.0)
        return self.store.upsert(MemoryItem(content=content, memory_type=memory_type, scope=scope, key=key, importance=importance, metadata=metadata or {}, expires_at=expires_at))

    def recall(self, scope: MemoryScope, query: str = "", *, memory_type: str | None = None, limit: int = 5) -> list[MemoryItem]:
        return self.store.search(scope, query, memory_type=memory_type, limit=limit)

    def forget(self, scope: MemoryScope, memory_id: str) -> None:
        self.store.delete(scope, memory_id)

    def extract_and_remember(self, scope: MemoryScope, messages: Iterable[dict[str, Any]], extractor: MemoryExtractor) -> list[MemoryItem]:
        saved: list[MemoryItem] = []
        for extracted in extractor.extract(messages):
            if self.policy.is_allowed(extracted.content, importance=extracted.importance, confidence=extracted.confidence):
                saved.append(self.remember(scope, extracted.content, memory_type=extracted.memory_type, key=extracted.key, importance=extracted.importance, metadata=extracted.metadata, expires_at=extracted.metadata.get("expires_at") if extracted.metadata else None))
        return saved

    def purge_expired(self, now: str | None = None) -> int:
        return self.store.purge_expired(now)


class MemoryWritePolicy:
    """Reject secrets, prompt-injection instructions and low-confidence facts."""

    _sensitive = re.compile(r"(?:api[_ -]?key|password|secret|token)\s*[:=]|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\b(?:\+?\d[\d -]{8,}\d)\b", re.IGNORECASE)
    _injection = re.compile(r"ignore\s+(?:all|previous|prior)\s+instructions|system\s+prompt|developer\s+message", re.IGNORECASE)

    def __init__(self, *, min_confidence: float = 0.6) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("minimum confidence must be between 0 and 1")
        self.min_confidence = min_confidence

    def is_allowed(self, content: str, *, importance: float, confidence: float) -> bool:
        return bool(content.strip()) and importance >= 0 and confidence >= self.min_confidence and not self._sensitive.search(content) and not self._injection.search(content)

    def validate(self, content: str, *, importance: float, confidence: float) -> None:
        if not self.is_allowed(content, importance=importance, confidence=confidence):
            raise ValueError("memory content rejected by safety policy")
