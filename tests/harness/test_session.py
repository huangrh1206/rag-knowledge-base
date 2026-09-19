from src.harness import (
    HarnessContextBuilder,
    HarnessEvent,
    SessionStore,
)


def test_session_context_accepts_langgraph_state() -> None:
    sessions = SessionStore()
    session = sessions.create("run-1")
    sessions.append("run-1", HarnessEvent("run-1", "input"))
    context = HarnessContextBuilder().build(
        session,
        state={"answer": "partial"},
        next_step=2,
    )

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
