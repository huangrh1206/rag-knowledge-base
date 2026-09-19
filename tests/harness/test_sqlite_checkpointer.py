from pathlib import Path
from tempfile import TemporaryDirectory

from src.harness import (
    HarnessEvent,
    HarnessPolicy,
    HarnessRuntime,
    SQLiteSessionStore,
    open_sqlite_checkpointer,
)


def test_sqlite_session_store_persists_audit_events() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        database = Path(directory) / "events.sqlite3"
        store = SQLiteSessionStore(database)
        store.create("run-1")
        store.append(
            "run-1",
            HarnessEvent("run-1", "tool_completed", {"result": "ok"}),
        )
        store.close()

        reopened = SQLiteSessionStore(database)
        session = reopened.get("run-1")
        assert session.events[0].payload == {"result": "ok"}
        reopened.close()


def test_langgraph_sqlite_checkpoint_survives_runtime_recreation() -> None:
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        database = Path(directory) / "checkpoints.sqlite3"
        steps = [
            lambda state: {**state, "count": state.get("count", 0) + 1},
            lambda state: {**state, "count": state["count"] + 1},
        ]
        first_store = SQLiteSessionStore(Path(directory) / "events.sqlite3")
        with open_sqlite_checkpointer(database) as first_checkpointer:
            paused = HarnessRuntime(
                sessions=first_store,
                policy=HarnessPolicy(max_steps=1),
                checkpointer=first_checkpointer,
            ).run("run-2", steps)
        first_store.close()

        second_store = SQLiteSessionStore(Path(directory) / "events.sqlite3")
        with open_sqlite_checkpointer(database) as second_checkpointer:
            resumed = HarnessRuntime(
                sessions=second_store,
                policy=HarnessPolicy(max_steps=2),
                checkpointer=second_checkpointer,
            ).run("run-2", steps, resume=True)

        assert paused.status == "paused"
        assert resumed.status == "completed"
        assert resumed.state == {"count": 2}
        second_store.close()
