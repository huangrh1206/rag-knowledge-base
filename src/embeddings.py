from collections.abc import Callable, Sequence
import time
from typing import Any

import numpy as np

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


class EmbeddingClient:
    def __init__(
        self,
        api: Any,
        model: str,
        batch_size: int = 64,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")

        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts <= 0
        ):
            raise ValueError("max_attempts must be a positive integer")

        self._api = api
        self._model = model
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._sleep = sleep

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        if not texts:
            raise ValueError("at least one text is required")

        vectors: list[list[float]] = []

        for offset in range(
            0,
            len(texts),
            self._batch_size,
        ):
            batch = list(
                texts[
                    offset : offset + self._batch_size
                ]
            )

            response = self._request(batch)

            ordered = sorted(
                response.data,
                key=lambda item: item.index,
            )

            if len(ordered) != len(batch):
                raise ValueError(
                    "embedding response count mismatch"
                )

            expected_indexes = list(
                range(len(batch))
            )
            actual_indexes = [
                item.index
                for item in ordered
            ]

            if actual_indexes != expected_indexes:
                raise ValueError(
                    "embedding response indexes mismatch"
                )

            vectors.extend(
                item.embedding
                for item in ordered
            )

        dimensions = {
            len(vector)
            for vector in vectors
        }

        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("embedding dimension mismatch")

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        return self.embed_texts([text])[0]

    def _request(
        self,
        batch: list[str],
    ) -> Any:
        retryable_errors = (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._api.create(
                    model=self._model,
                    input=batch,
                )
            except retryable_errors:
                if attempt == self._max_attempts:
                    raise
                delay = 2 ** (attempt - 1)
                self._sleep(delay)

        raise RuntimeError(
            "embedding retry loop ended unexpectedly"
        )
