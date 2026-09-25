"""Batch runner for deterministic Agent Harness evaluations."""

from dataclasses import dataclass
from typing import Callable, Iterable
from uuid import uuid4

from src.agent.evaluation import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    AgentEvaluator,
)
from src.harness import HarnessResult


EvaluationRunner = Callable[[str, dict], HarnessResult]


@dataclass(frozen=True)
class AgentEvaluationReport:
    case_count: int
    task_success_rate: float
    average_tool_selection_accuracy: float
    average_tool_error_rate: float
    human_intervention_rate: float
    results: tuple[AgentEvaluationResult, ...]


class EvaluationHarness:
    def __init__(
        self,
        runner: EvaluationRunner,
        evaluator: AgentEvaluator | None = None,
    ) -> None:
        self._runner = runner
        self._evaluator = evaluator or AgentEvaluator()

    def run(
        self,
        cases: Iterable[AgentEvaluationCase],
    ) -> AgentEvaluationReport:
        case_values = tuple(cases)
        if not case_values:
            raise ValueError("evaluation requires at least one case")
        results = tuple(
            self._run_case(case)
            for case in case_values
        )
        count = len(results)
        return AgentEvaluationReport(
            case_count=count,
            task_success_rate=sum(item.passed for item in results) / count,
            average_tool_selection_accuracy=sum(
                item.tool_selection_accuracy for item in results
            ) / count,
            average_tool_error_rate=sum(
                item.tool_error_rate for item in results
            ) / count,
            human_intervention_rate=sum(
                item.human_interventions > 0 for item in results
            ) / count,
            results=results,
        )

    def _run_case(self, case: AgentEvaluationCase) -> AgentEvaluationResult:
        run_id = f"eval-{case.case_id}-{uuid4().hex}"
        result = self._runner(run_id, dict(case.input))
        return self._evaluator.evaluate(case, result)
