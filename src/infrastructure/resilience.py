"""Small synchronous rate-limit and circuit-breaker primitives."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any


class RateLimitExceeded(RuntimeError):
    pass


class CircuitOpenError(RuntimeError):
    pass


class SlidingWindowRateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("rate limit and window must be positive")
        self._limit = limit
        self._window = window_seconds
        self._clock = clock or time.monotonic
        self._requests: dict[str, deque[float]] = {}

    def acquire(self, key: str) -> None:
        now = self._clock()
        requests = self._requests.setdefault(key, deque())
        while requests and now - requests[0] >= self._window:
            requests.popleft()
        if len(requests) >= self._limit:
            raise RateLimitExceeded(f"rate limit exceeded: {key}")
        requests.append(now)


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0 or self.recovery_seconds <= 0:
            raise ValueError("circuit breaker limits must be positive")
        self._failures = 0
        self._opened_at: float | None = None

    def call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        now = self.clock()
        if self._opened_at is not None:
            if now - self._opened_at < self.recovery_seconds:
                raise CircuitOpenError("circuit breaker is open")
            self._opened_at = None
            self._failures = 0
        try:
            value = operation(*args, **kwargs)
        except Exception:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = now
            raise
        self._failures = 0
        return value

    def call_with_fallback(
        self,
        operation: Callable[[], Any],
        fallback: Callable[[Exception], Any],
    ) -> Any:
        """Return a controlled fallback for open circuits or call failures."""

        try:
            return self.call(operation)
        except Exception as exc:
            return fallback(exc)
