from evaluation.harness import EvaluationHarness
from src.agent.evaluation import AgentEvaluationCase, ExpectedStateGrader
from src.harness import HarnessResult


def test_evaluation_harness_uses_isolated_run_ids_and_aggregates() -> None:
    run_ids = []

    def runner(run_id: str, input_value: dict) -> HarnessResult:
        run_ids.append(run_id)
        return HarnessResult(
            run_id,
            "completed",
            {"answer": input_value["expected"]},
            1,
            (),
        )

    cases = [
        AgentEvaluationCase(
            case_id="one",
            input={"expected": "a"},
            outcome_grader=ExpectedStateGrader({"answer": "a"}),
        ),
        AgentEvaluationCase(
            case_id="two",
            input={"expected": "b"},
            outcome_grader=ExpectedStateGrader({"answer": "b"}),
        ),
    ]

    report = EvaluationHarness(runner).run(cases)

    assert len(set(run_ids)) == 2
    assert report.case_count == 2
    assert report.task_success_rate == 1.0
