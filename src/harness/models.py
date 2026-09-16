"""Data models for resumable Harness runs."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class HarnessEvent:
    """One immutable entry in a run's execution trace."""

    run_id: str
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class HarnessCheckpoint:
    run_id: str
    next_step: int
    state: dict[str, Any]
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class HarnessResult:
    run_id: str
    status: str
    state: dict[str, Any]
    next_step: int
    events: tuple[HarnessEvent, ...]


@dataclass
class HarnessSession:
    run_id: str
    events: list[HarnessEvent] = field(default_factory=list)
    checkpoint: HarnessCheckpoint | None = None

    def append(self, event: HarnessEvent) -> None:
        self.events.append(event)

    def record(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> HarnessEvent:
        event = HarnessEvent(
            run_id=self.run_id,
            event_type=event_type,
            payload=payload or {},
        )
        self.append(event)
        return event
