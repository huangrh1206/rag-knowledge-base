from src.infrastructure.observability import (
    ObservabilityRecorder,
    TraceContext,
    summarize_events,
)


def test_recorder_adds_trace_context_and_redacts_payload() -> None:
    events = []
    context = TraceContext.create(run_id="run-1", request_id="request-1")
    recorder = ObservabilityRecorder(events.append, context=context)

    recorder(
        "tool_requested",
        {
            "name": "search",
            "api_key": "secret",
            "query": "safe",
            "arguments": '{"token":"secret"}',
        },
    )

    event = events[0]
    assert event.trace_id == context.trace_id
    assert event.payload == {
        "name": "search",
        "api_key": "[REDACTED]",
        "query": "safe",
        "arguments": "[OMITTED]",
    }


def test_event_summary_calculates_agent_reliability_metrics() -> None:
    events = []
    recorder = ObservabilityRecorder(
        events.append,
        context=TraceContext.create(run_id="run-2"),
        clock=iter([1.0, 2.0, 3.0, 5.0]).__next__,
    )
    recorder("model_requested", {"round": 1})
    recorder("tool_requested", {"name": "search"})
    recorder("tool_failed", {"name": "search", "error": "timeout"})
    recorder("run_interrupted", {"next_step": 1})

    summary = summarize_events(events)

    assert summary.model_calls == 1
    assert summary.tool_calls == 1
    assert summary.tool_errors == 1
    assert summary.tool_error_rate == 1.0
    assert summary.human_interventions == 1
    assert summary.elapsed_ms == 4000.0
    assert summary.token_count == 0
    assert summary.estimated_cost == 0.0
