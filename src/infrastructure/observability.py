"""Structured, redacted events and deterministic run metrics."""

from dataclasses import dataclass
import time
from typing import Any, Callable, Iterable
from uuid import uuid4

from src.authz import redact_sensitive


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    run_id: str
    request_id: str

    @classmethod
    def create(
        cls,
        run_id: str,
        request_id: str | None = None,
    ) -> "TraceContext":
        return cls(
            trace_id=uuid4().hex,
            run_id=run_id,
            request_id=request_id or uuid4().hex,
        )


@dataclass(frozen=True)
class ObservabilityEvent:
    event_type: str
    payload: dict[str, Any]
    trace_id: str
    run_id: str
    request_id: str
    timestamp: float


@dataclass(frozen=True)
class RunMetrics:
    model_calls: int
    tool_calls: int
    tool_errors: int
    tool_error_rate: float
    human_interventions: int
    elapsed_ms: float
    token_count: int
    estimated_cost: float


EventSink = Callable[[ObservabilityEvent], None]


class ObservabilityRecorder:
    """Adapt Agent event callbacks to structured and redacted events."""

    def __init__(
        self,
        sink: EventSink,
        *,
        context: TraceContext,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._sink = sink
        self.context = context
        self._clock = clock or time.time

    def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        self._sink(
            ObservabilityEvent(
                event_type=event_type,
                payload=_sanitize_payload(payload),
                trace_id=self.context.trace_id,
                run_id=self.context.run_id,
                request_id=self.context.request_id,
                timestamp=self._clock(),
            )
        )


def summarize_events(events: Iterable[ObservabilityEvent]) -> RunMetrics:
    values = tuple(events)
    model_calls = sum(event.event_type == "model_requested" for event in values)
    tool_calls = sum(event.event_type == "tool_requested" for event in values)
    tool_errors = sum(event.event_type == "tool_failed" for event in values)
    interventions = sum(
        event.event_type in {"approval_requested", "run_interrupted"}
        for event in values
    )
    token_count = sum(
        _numeric(event.payload.get("token_count")) for event in values
    )
    estimated_cost = sum(
        float(_numeric(event.payload.get("cost"))) for event in values
    )
    elapsed_ms = 0.0
    if len(values) >= 2:
        elapsed_ms = max(0.0, values[-1].timestamp - values[0].timestamp) * 1000
    return RunMetrics(
        model_calls=model_calls,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        tool_error_rate=tool_errors / tool_calls if tool_calls else 0.0,
        human_interventions=interventions,
        elapsed_ms=elapsed_ms,
        token_count=int(token_count),
        estimated_cost=estimated_cost,
    )


def _sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = redact_sensitive(payload)
    for key in ("arguments", "content", "result"):
        if key in sanitized and sanitized[key] not in (None, ""):
            sanitized[key] = "[OMITTED]"
    return sanitized


def _numeric(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)
