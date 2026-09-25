"""Deterministic transcript and outcome evaluation for Agent runs."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.harness import HarnessResult


class OutcomeGrader(Protocol):
    def grade(self, result: HarnessResult) -> bool:
        ...


@dataclass(frozen=True)
class ExpectedStateGrader:
    expected: dict[str, Any]

    def grade(self, result: HarnessResult) -> bool:
        return all(result.state.get(key) == value for key, value in self.expected.items())


@dataclass(frozen=True)
class AgentEvaluationCase:
    case_id: str
    input: dict[str, Any] = field(default_factory=dict)
    expected_tools: frozenset[str] = frozenset()
    outcome_grader: OutcomeGrader | None = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("evaluation case id cannot be empty")


@dataclass(frozen=True)
class AgentEvaluationResult:
    case_id: str
    passed: bool
    transcript_passed: bool
    outcome_passed: bool
    tool_selection_accuracy: float
    tool_error_rate: float
    rounds: int
    human_interventions: int


class AgentEvaluator:
    def evaluate(
        self,
        case: AgentEvaluationCase,
        result: HarnessResult,
    ) -> AgentEvaluationResult:
        requested_tools = {
            str(event.payload.get("name"))
            for event in result.events
            if event.event_type == "tool_requested"
        }
        expected = case.expected_tools
        transcript_passed = not expected or requested_tools == expected
        selection_accuracy = _set_accuracy(expected, requested_tools)
        tool_calls = sum(
            event.event_type == "tool_requested" for event in result.events
        )
        tool_errors = sum(
            event.event_type == "tool_failed" for event in result.events
        )
        outcome_passed = (
            result.status == "completed"
            if case.outcome_grader is None
            else case.outcome_grader.grade(result)
        )
        rounds = sum(
            event.event_type == "model_requested" for event in result.events
        )
        interventions = sum(
            event.event_type in {"approval_requested", "run_interrupted"}
            for event in result.events
        )
        return AgentEvaluationResult(
            case_id=case.case_id,
            passed=transcript_passed and outcome_passed,
            transcript_passed=transcript_passed,
            outcome_passed=outcome_passed,
            tool_selection_accuracy=selection_accuracy,
            tool_error_rate=tool_errors / tool_calls if tool_calls else 0.0,
            rounds=rounds,
            human_interventions=interventions,
        )


def _set_accuracy(expected: frozenset[str], actual: set[str]) -> float:
    if not expected and not actual:
        return 1.0
    union = expected | actual
    return len(expected & actual) / len(union) if union else 1.0
