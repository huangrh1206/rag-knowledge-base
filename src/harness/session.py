"""In-memory append-only session store."""

from src.harness.models import HarnessCheckpoint, HarnessEvent, HarnessSession


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, HarnessSession] = {}

    def create(self, run_id: str) -> HarnessSession:
        if run_id in self._sessions:
            raise ValueError(f"run already exists: {run_id}")
        session = HarnessSession(run_id=run_id)
        self._sessions[run_id] = session
        return session

    def get(self, run_id: str) -> HarnessSession:
        try:
            return self._sessions[run_id]
        except KeyError as exc:
            raise KeyError(f"run not found: {run_id}") from exc

    def append(self, run_id: str, event: HarnessEvent) -> None:
        self.get(run_id).append(event)

    def save_checkpoint(self, checkpoint: HarnessCheckpoint) -> None:
        self.get(checkpoint.run_id).checkpoint = checkpoint
