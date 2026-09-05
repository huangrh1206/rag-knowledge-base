"""Checkpoint access kept separate from event storage."""

from src.harness.models import HarnessCheckpoint
from src.harness.session import SessionStore


class CheckpointStore:
    def __init__(self, sessions: SessionStore) -> None:
        self._sessions = sessions

    def save(self, checkpoint: HarnessCheckpoint) -> None:
        self._sessions.save_checkpoint(checkpoint)

    def load(self, run_id: str) -> HarnessCheckpoint | None:
        return self._sessions.get(run_id).checkpoint
