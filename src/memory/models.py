"""Data models shared by memory implementations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MemoryScope:
    """Isolation boundary for a memory item."""

    tenant_id: str
    user_id: str
    scene_id: str = "default"

    def __post_init__(self) -> None:
        if not self.tenant_id or not self.user_id or not self.scene_id:
            raise ValueError("memory scope fields must be non-empty")


@dataclass
class MemoryItem:
    content: str
    memory_type: str
    scope: MemoryScope
    key: str | None = None
    importance: float = 0.5
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    expires_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("memory content must be non-empty")
        if not self.memory_type.strip():
            raise ValueError("memory type must be non-empty")
        if not 0 <= self.importance <= 1:
            raise ValueError("importance must be between 0 and 1")

    def is_expired(self, now: str | None = None) -> bool:
        return self.expires_at is not None and self.expires_at <= (now or utc_now())
