"""Build a bounded view of a Harness session and graph state."""

from typing import Any

from src.harness.models import HarnessSession


class HarnessContextBuilder:
    def build(
        self,
        session: HarnessSession,
        state: dict[str, Any] | None = None,
        next_step: int = 0,
    ) -> dict[str, Any]:
        return {
            "run_id": session.run_id,
            "next_step": next_step,
            "state": dict(state or {}),
            "event_count": len(session.events),
        }
