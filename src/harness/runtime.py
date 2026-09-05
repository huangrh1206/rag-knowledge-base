"""Resumable step runner for Agent, Workflow, and MCP orchestration."""

from collections.abc import Callable, Sequence
from typing import Any

from src.harness.checkpoint import CheckpointStore
from src.harness.models import (
    HarnessCheckpoint,
    HarnessEvent,
    HarnessResult,
)
from src.harness.policy import HarnessPolicy
from src.harness.session import SessionStore

Step = Callable[[dict[str, Any]], dict[str, Any]]


class HarnessRuntime:
    def __init__(
        self,
        sessions: SessionStore | None = None,
        policy: HarnessPolicy | None = None,
    ) -> None:
        self.sessions = sessions or SessionStore()
        self.checkpoints = CheckpointStore(self.sessions)
        self.policy = policy or HarnessPolicy()

    def run(
        self,
        run_id: str,
        steps: Sequence[Step],
        *,
        resume: bool = False,
        cancel_at: int | None = None,
    ) -> HarnessResult:
        session = (
            self.sessions.get(run_id)
            if resume
            else self.sessions.create(run_id)
        ) # 若resume=False，需要一个新的run_id
        checkpoint = self.checkpoints.load(run_id) if resume else None
        state = dict(checkpoint.state) if checkpoint else {}
        next_step = checkpoint.next_step if checkpoint else 0
        self.sessions.append(
            run_id,
            HarnessEvent(
                run_id,
                "run_started",
                {"next_step": next_step, "state": dict(state)},
            ),
        )

        status = "completed"
        while next_step < len(steps):
            if next_step >= self.policy.max_steps:
                status = "paused"
                break
            if cancel_at is not None and next_step >= cancel_at:
                status = "cancelled"
                break
            step_input = dict(state)
            self.sessions.append(
                run_id,
                HarnessEvent(
                    run_id,
                    "step_started",
                    {"step": next_step, "state": step_input},
                ),
            )
            try:
                state = dict(steps[next_step](step_input))
            except Exception as exc:
                status = "failed"
                self.sessions.append(
                    run_id,
                    HarnessEvent(
                        run_id,
                        "step_failed",
                        {
                            "step": next_step,
                            "error": str(exc),
                            "state": step_input,
                        },
                    ),
                )
                break
            next_step += 1
            self.sessions.append(
                run_id,
                HarnessEvent(
                    run_id,
                    "step_succeeded",
                    {
                        "step": next_step - 1,
                        "state": dict(state),
                    },
                ),
            )
            self.checkpoints.save(
                HarnessCheckpoint(run_id, next_step, dict(state))
            )

        self.sessions.append(
            run_id,
            HarnessEvent(run_id, "run_finished", {"status": status}),
        )
        return HarnessResult(
            run_id,
            status,
            state,
            next_step,
            tuple(session.events),
        )
