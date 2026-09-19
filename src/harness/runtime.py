"""LangGraph-backed execution runtime for Harness workflows."""

from collections.abc import Sequence
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.harness.models import HarnessEvent, HarnessResult, HarnessSession
from src.harness.policy import HarnessPolicy
from src.harness.session import SessionStore
from src.harness.store import HarnessSessionStore
from src.harness.tracing import HarnessEventRecorder


_NO_RESUME_VALUE = object()


class HarnessRuntime:
    """Execute Harness steps as a checkpointed LangGraph.

    ``run_id`` maps directly to LangGraph's ``thread_id``.  Consequently a
    new runtime instance can resume a run when it is given the same persistent
    checkpointer (for example ``SqliteSaver``).
    """

    def __init__(
        self,
        sessions: HarnessSessionStore | None = None,
        policy: HarnessPolicy | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.sessions = sessions or SessionStore()
        self.policy = policy or HarnessPolicy()
        self.checkpointer = checkpointer or InMemorySaver()

    def recorder(self, run_id: str) -> HarnessEventRecorder:
        """Return an Agent/Tool event callback bound to a run."""

        return HarnessEventRecorder(self.sessions, run_id)

    def run(
        self,
        run_id: str,
        steps: Sequence[Any],
        *,
        resume: bool = False,
        resume_value: Any = _NO_RESUME_VALUE,
        cancel_at: int | None = None,
    ) -> HarnessResult:
        if not steps:
            raise ValueError("at least one step is required")

        self._get_or_create_session(run_id, resume)
        config = {"configurable": {"thread_id": run_id}}
        graph_builder = self._build_graph(steps, run_id)

        # Read the persisted cursor before compiling this invocation's budget.
        base_graph = graph_builder.compile(checkpointer=self.checkpointer)
        snapshot = base_graph.get_state(config)
        start = self._next_step(snapshot, len(steps)) if resume else 0

        if cancel_at is not None and cancel_at <= start:
            self._record(run_id, "run_cancelled", {"next_step": start})
            return self._result(run_id, "cancelled", snapshot.values, start)

        stop_index = min(start + self.policy.max_steps - 1, len(steps) - 1)
        interrupt_after = (
            [f"step_{stop_index}"] if stop_index < len(steps) - 1 else None
        )
        interrupt_before = (
            [f"step_{cancel_at}"]
            if cancel_at is not None and start <= cancel_at < len(steps)
            else None
        )
        graph = graph_builder.compile(
            checkpointer=self.checkpointer,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
        )

        self._record(run_id, "run_started", {"resume": resume, "next_step": start})
        invoke_input: Any = {}
        if resume:
            invoke_input = (
                Command(resume=resume_value)
                if resume_value is not _NO_RESUME_VALUE
                else None
            )
        try:
            graph.invoke(invoke_input, config)
        except Exception as exc:
            latest = graph.get_state(config)
            self._record(run_id, "run_failed", {"error": str(exc)})
            return self._result(
                run_id,
                "failed",
                latest.values,
                self._next_step(latest, len(steps)),
            )

        latest = graph.get_state(config)
        next_step = self._next_step(latest, len(steps))
        status = "completed"
        if cancel_at is not None and latest.next == (f"step_{cancel_at}",):
            status = "cancelled"
            self._record(run_id, "run_cancelled", {"next_step": next_step})
        elif any(task.interrupts for task in latest.tasks):
            status = "interrupted"
            self._record(run_id, "run_interrupted", {"next_step": next_step})
        elif latest.next:
            status = "paused"
            self._record(run_id, "run_paused", {"next_step": next_step})
        else:
            self._record(run_id, "run_completed", {"state": dict(latest.values)})

        return self._result(run_id, status, latest.values, next_step)

    def _build_graph(
        self,
        steps: Sequence[Any],
        run_id: str,
    ) -> StateGraph:
        builder = StateGraph(dict)
        for index, step in enumerate(steps):
            name = f"step_{index}"

            def node(
                state: dict[str, Any],
                step: Any = step,
                index: int = index,
                node_name: str = name,
            ) -> dict[str, Any]:
                self._record(
                    run_id,
                    "step_started",
                    {
                        "step": index,
                        "name": getattr(step, "name", node_name),
                    },
                )
                try:
                    result = step(dict(state))
                    if not isinstance(result, dict):
                        raise TypeError("Harness step must return a dictionary")
                    self._record(
                        run_id,
                        "step_succeeded",
                        {"step": index, "state": dict(result)},
                    )
                    return result
                except Exception as exc:
                    self._record(
                        run_id,
                        "step_failed",
                        {"step": index, "error": str(exc)},
                    )
                    raise

            builder.add_node(name, node)
            if index == 0:
                builder.add_edge(START, name)
            else:
                builder.add_edge(f"step_{index - 1}", name)
        builder.add_edge(f"step_{len(steps) - 1}", END)
        return builder

    def _get_or_create_session(self, run_id: str, resume: bool) -> HarnessSession:
        if not resume:
            return self.sessions.create(run_id)
        try:
            return self.sessions.get(run_id)
        except KeyError:
            return self.sessions.create(run_id)

    def _record(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.sessions.append(
            run_id,
            HarnessEvent(run_id, event_type, payload or {}),
        )

    @staticmethod
    def _next_step(snapshot: Any, step_count: int) -> int:
        if not snapshot.next:
            return step_count
        name = snapshot.next[0]
        return int(name.removeprefix("step_"))

    def _result(
        self,
        run_id: str,
        status: str,
        state: Any,
        next_step: int,
    ) -> HarnessResult:
        session = self.sessions.get(run_id)
        return HarnessResult(run_id, status, dict(state), next_step, tuple(session.events))
