import pytest

from src.infrastructure import (
    CircuitBreaker,
    CircuitOpenError,
    RateLimitExceeded,
    SlidingWindowRateLimiter,
)


def test_rate_limiter_releases_capacity_after_window() -> None:
    times = iter([0.0, 0.5, 2.0])
    limiter = SlidingWindowRateLimiter(1, 1.0, clock=times.__next__)

    limiter.acquire("tenant-a")
    with pytest.raises(RateLimitExceeded):
        limiter.acquire("tenant-a")
    limiter.acquire("tenant-a")


def test_circuit_breaker_opens_and_recovers() -> None:
    now = [0.0]
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_seconds=5.0,
        clock=lambda: now[0],
    )

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("failed")))
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "blocked")

    now[0] = 5.0
    assert breaker.call(lambda: "recovered") == "recovered"


def test_circuit_breaker_supports_controlled_fallback() -> None:
    breaker = CircuitBreaker(failure_threshold=1)

    value = breaker.call_with_fallback(
        lambda: (_ for _ in ()).throw(TimeoutError("slow")),
        lambda error: f"fallback: {type(error).__name__}",
    )

    assert value == "fallback: TimeoutError"
