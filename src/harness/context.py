"""Build bounded execution context from a session checkpoint."""

from typing import Any

from src.harness.models import HarnessCheckpoint, HarnessSession


class HarnessContextBuilder:
    def build(
        self,
        session: HarnessSession,
        checkpoint: HarnessCheckpoint | None = None,
    ) -> dict[str, Any]:
        active = checkpoint or session.checkpoint
        return {
            "run_id": session.run_id,
            "next_step": active.next_step if active else 0,
            "state": dict(active.state) if active else {},
            "event_count": len(session.events),
        }
