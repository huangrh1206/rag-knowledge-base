from __future__ import annotations
from dataclasses import dataclass
import asyncio
import random
from typing import Any, Callable

@dataclass(frozen=True)
class TaskPolicy:
    max_retries: int = 2
    step_timeout: float = 30.0
    task_timeout: float | None = None
    backoff_base: float = 0.1
    max_supersteps: int = 100
    def __post_init__(self):
        if self.max_retries < 0 or self.step_timeout <= 0 or self.backoff_base < 0 or self.max_supersteps <= 0: raise ValueError("invalid task policy")

def is_retryable_error(error: BaseException) -> bool:
    return bool(getattr(error, "retryable", False) or isinstance(error, (TimeoutError, asyncio.TimeoutError, ConnectionError)))

async def retry_async(operation: Callable[[], Any], *, policy: TaskPolicy, on_retry: Callable[[int], None] | None = None) -> Any:
    last: BaseException | None = None
    for attempt in range(policy.max_retries + 1):
        try: return await asyncio.wait_for(operation(), policy.step_timeout)
        except Exception as exc:
            last = exc
            if attempt >= policy.max_retries or not is_retryable_error(exc): raise
            if on_retry: on_retry(attempt + 1)
            await asyncio.sleep(policy.backoff_base * (2 ** attempt) + random.random() * policy.backoff_base / 10)
    raise last or RuntimeError("retry failed")

