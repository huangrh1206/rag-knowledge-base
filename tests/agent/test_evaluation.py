from src.agent.evaluation import (
    AgentEvaluationCase,
    AgentEvaluator,
    ExpectedStateGrader,
)
from src.harness import HarnessEvent, HarnessResult


def result(*, state: dict, events: tuple[HarnessEvent, ...]) -> HarnessResult:
    return HarnessResult("run", "completed", state, 1, events)


def test_evaluator_separates_transcript_and_outcome_grades() -> None:
    case = AgentEvaluationCase(
        case_id="case-1",
        expected_tools=frozenset({"search"}),
        outcome_grader=ExpectedStateGrader({"ticket_status": "closed"}),
    )
    evaluation = AgentEvaluator().evaluate(
        case,
        result(
            state={"ticket_status": "open"},
            events=(
                HarnessEvent("run", "tool_requested", {"name": "search"}),
                HarnessEvent("run", "run_completed", {}),
            ),
        ),
    )

    assert evaluation.transcript_passed is True
    assert evaluation.outcome_passed is False
    assert evaluation.passed is False


def test_evaluator_reports_tool_selection_and_failure_metrics() -> None:
    case = AgentEvaluationCase(
        case_id="case-2",
        expected_tools=frozenset({"search"}),
        outcome_grader=ExpectedStateGrader({"answer": "done"}),
    )
    evaluation = AgentEvaluator().evaluate(
        case,
        result(
            state={"answer": "done"},
            events=(
                HarnessEvent("run", "tool_requested", {"name": "other"}),
                HarnessEvent("run", "tool_failed", {"name": "other"}),
            ),
        ),
    )

    assert evaluation.tool_selection_accuracy == 0.0
    assert evaluation.tool_error_rate == 1.0
    assert evaluation.outcome_passed is True
