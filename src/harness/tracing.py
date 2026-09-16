"""Event callback adapter for Agent and tool execution."""

from typing import Any

from src.harness.models import HarnessEvent
from src.harness.store import HarnessSessionStore


class HarnessEventRecorder:
    def __init__(self, sessions: HarnessSessionStore, run_id: str) -> None:
        self._sessions = sessions
        self._run_id = run_id

    def __call__(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self._sessions.append(
            self._run_id,
            HarnessEvent(self._run_id, event_type, dict(payload)),
        )
