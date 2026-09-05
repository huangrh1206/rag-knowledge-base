import pytest

from src.harness import HarnessPolicy, HarnessRuntime


def test_runtime_pauses_at_budget_and_resumes_from_checkpoint() -> None:
    runtime = HarnessRuntime(policy=HarnessPolicy(max_steps=2))
    steps = [
        lambda state: {"count": state.get("count", 0) + 1},
        lambda state: {"count": state["count"] + 1},
        lambda state: {"count": state["count"] + 1},
    ]

    paused = runtime.run("run-1", steps)
    resumed = HarnessRuntime(
        sessions=runtime.sessions,
        policy=HarnessPolicy(max_steps=5),
    ).run("run-1", steps, resume=True)

    assert paused.status == "paused"
    assert paused.state == {"count": 2}
    assert resumed.status == "completed"
    assert resumed.state == {"count": 3}
    event_types = [event.event_type for event in resumed.events]
    assert "step_started" in event_types
    success = next(
        event
        for event in resumed.events
        if event.event_type == "step_succeeded"
    )
    assert "state" in success.payload


def test_runtime_records_failure_and_supports_cancellation() -> None:
    runtime = HarnessRuntime()
    failed = runtime.run("failed", [lambda _: 1])
    cancelled = runtime.run("cancelled", [lambda _: {}, lambda _: {}], cancel_at=0)

    assert failed.status == "failed"
    assert any(event.event_type == "step_failed" for event in failed.events)
    assert cancelled.status == "cancelled"


def test_policy_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError):
        HarnessPolicy(max_steps=0)
