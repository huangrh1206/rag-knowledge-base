from src.harness import (
    HarnessCheckpoint,
    HarnessContextBuilder,
    HarnessEvent,
    SessionStore,
)


def test_session_and_context_restore_checkpoint_state() -> None:
    sessions = SessionStore()
    session = sessions.create("run-1")
    sessions.append("run-1", HarnessEvent("run-1", "input"))
    checkpoint = HarnessCheckpoint("run-1", 2, {"answer": "partial"})
    sessions.save_checkpoint(checkpoint)

    context = HarnessContextBuilder().build(session)

    assert context == {
        "run_id": "run-1",
        "next_step": 2,
        "state": {"answer": "partial"},
        "event_count": 1,
    }


def test_session_record_creates_structured_event() -> None:
    sessions = SessionStore()
    session = sessions.create("run-2")

    event = session.record(
        "model_completed",
        {"response": "answer", "token_count": 12},
    )

    assert event in session.events
    assert event.run_id == "run-2"
