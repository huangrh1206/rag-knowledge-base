"""Input validation for common prompt-injection instructions."""

import re

from src.agent.types import AgentValidationError


class PromptInjectionGuard:
    def __init__(self, patterns: tuple[str, ...] | None = None) -> None:
        raw_patterns = patterns or (
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"reveal\s+(the\s+)?system\s+prompt",
            r"show\s+(me\s+)?(your\s+)?hidden\s+instructions",
        )
        self._patterns = tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in raw_patterns
        )

    def validate(self, value: str) -> None:
        if any(pattern.search(value) for pattern in self._patterns):
            raise AgentValidationError("possible prompt injection detected")
