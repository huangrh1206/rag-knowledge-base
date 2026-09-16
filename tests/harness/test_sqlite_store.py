from pathlib import Path
from tempfile import TemporaryDirectory

from src.harness import (
    HarnessCheckpoint,
    HarnessEvent,
    HarnessPolicy,
    HarnessRuntime,
    SQLiteSessionStore,
)


def test_sqlite_store_survives_reopening() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        database = Path(directory) / "harness.sqlite3"
        store = SQLiteSessionStore(database)
        store.create("run-1")
        store.append(
            "run-1",
            HarnessEvent("run-1", "tool_completed", {"result": "ok"}),
        )
        store.save_checkpoint(
            HarnessCheckpoint("run-1", 1, {"answer": "partial"})
        )
        store.close()

        reopened = SQLiteSessionStore(database)
        session = reopened.get("run-1")

        assert session.events[0].payload == {"result": "ok"}
        assert session.checkpoint is not None
        assert session.checkpoint.next_step == 1
        assert session.checkpoint.state == {"answer": "partial"}
        reopened.close()


def test_runtime_resumes_with_a_new_sqlite_store() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        database = Path(directory) / "resume.sqlite3"
        steps = [
            lambda state: {
                **state,
                "count": state.get("count", 0) + 1,
            },
            lambda state: {**state, "count": state["count"] + 1},
        ]
        first_store = SQLiteSessionStore(database)
        paused = HarnessRuntime(
            sessions=first_store,
            policy=HarnessPolicy(max_steps=1),
        ).run("run-2", steps)
        first_store.close()

        second_store = SQLiteSessionStore(database)
        resumed = HarnessRuntime(
            sessions=second_store,
            policy=HarnessPolicy(max_steps=2),
        ).run("run-2", steps, resume=True)

        assert paused.status == "paused"
        assert resumed.status == "completed"
        assert resumed.state == {"count": 2}
        assert len(resumed.events) > len(paused.events)
        second_store.close()
