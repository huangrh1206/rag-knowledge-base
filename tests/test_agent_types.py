import pytest

from src.agent_types import (
    AgentRunConfig,
    AgentValidationError,
)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_rounds", 0, "max rounds"),
        ("max_tool_calls", 0, "max tool calls"),
        ("max_elapsed_seconds", 0, "max elapsed"),
    ],
)
def test_run_config_rejects_non_positive_limits(
    field: str,
    value: int,
    message: str,
) -> None:
    values = {
        "max_rounds": 3,
        "max_tool_calls": 10,
        "max_elapsed_seconds": 60,
    }
    values[field] = value

    with pytest.raises(AgentValidationError, match=message):
        AgentRunConfig(**values)


def test_run_config_has_conservative_defaults() -> None:
    config = AgentRunConfig()

    assert config.max_rounds == 3
    assert config.max_tool_calls == 10
    assert config.max_elapsed_seconds == 60.0
