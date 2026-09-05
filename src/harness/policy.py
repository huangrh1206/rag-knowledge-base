"""Execution limits for Harness runs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessPolicy:
    max_steps: int = 20

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
