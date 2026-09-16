"""Storage contract shared by in-memory and persistent Harness stores."""

from typing import Protocol

from src.harness.models import HarnessCheckpoint, HarnessEvent, HarnessSession


class HarnessSessionStore(Protocol):
    def create(self, run_id: str) -> HarnessSession:
        ...

    def get(self, run_id: str) -> HarnessSession:
        ...

    def append(self, run_id: str, event: HarnessEvent) -> None:
        ...

    def save_checkpoint(self, checkpoint: HarnessCheckpoint) -> None:
        ...
