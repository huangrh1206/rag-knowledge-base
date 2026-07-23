from types import SimpleNamespace

import httpx
import numpy as np
from openai import AuthenticationError, RateLimitError
import pytest

from src.embeddings import EmbeddingClient


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def create(
        self,
        *,
        model: str,
        input: list[str],
    ) -> SimpleNamespace:
        self.calls += 1
        data = [
            SimpleNamespace(
                index=index,
                embedding=[float(index), 1.0],
            )
            for index, _ in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)))


def make_rate_limit_error() -> RateLimitError:
    request = httpx.Request(
        "POST",
        "https://example.test/embeddings",
    )
    response = httpx.Response(
        429,
        request=request,
    )
    return RateLimitError(
        "rate limit",
        response=response,
        body=None,
    )


def make_authentication_error() -> AuthenticationError:
    request = httpx.Request(
        "POST",
        "https://example.test/embeddings",
    )
    response = httpx.Response(
        401,
        request=request,
    )
    return AuthenticationError(
        "invalid API key",
        response=response,
        body=None,
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("batch_size", 0),
        ("max_attempts", 0),
    ],
)
def test_embedding_client_rejects_nonpositive_limits(
    keyword: str,
    value: int,
) -> None:
    arguments = {
        "api": FakeEmbeddings(),
        "model": "embedding-model",
        keyword: value,
    }

    with pytest.raises(ValueError, match=keyword):
        EmbeddingClient(**arguments)


def test_embed_texts_rejects_empty_input() -> None:
    client = EmbeddingClient(
        FakeEmbeddings(),
        model="embedding-model",
        sleep=lambda _: None,
    )

    with pytest.raises(ValueError, match="at least one text"):
        client.embed_texts([])


def test_embed_texts_batches_requests_and_restores_index_order() -> None:
    api = FakeEmbeddings()
    client = EmbeddingClient(
        api,
        model="embedding-model",
        batch_size=2,
        sleep=lambda _: None,
    )

    result = client.embed_texts(["a", "b", "c"])

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert api.calls == 2


def test_embed_query_returns_one_dimensional_vector() -> None:
    api = FakeEmbeddings()
    client = EmbeddingClient(
        api,
        model="embedding-model",
        sleep=lambda _: None,
    )

    result = client.embed_query("question")

    np.testing.assert_array_equal(
        result,
        np.array([0.0, 1.0], dtype=np.float32),
    )
    assert result.ndim == 1
    assert api.calls == 1


def test_embed_texts_retries_rate_limits_with_exponential_backoff() -> None:
    class FlakyEmbeddings:
        def __init__(self) -> None:
            self.calls = 0

        def create(
            self,
            *,
            model: str,
            input: list[str],
        ) -> SimpleNamespace:
            self.calls += 1
            if self.calls < 3:
                raise make_rate_limit_error()
            item = SimpleNamespace(
                index=0,
                embedding=[1.0, 2.0],
            )
            return SimpleNamespace(data=[item])

    api = FlakyEmbeddings()
    delays: list[float] = []
    client = EmbeddingClient(
        api,
        model="embedding-model",
        max_attempts=3,
        sleep=delays.append,
    )

    result = client.embed_texts(["a"])

    np.testing.assert_array_equal(
        result,
        np.array([[1.0, 2.0]], dtype=np.float32),
    )
    assert api.calls == 3
    assert delays == [1, 2]


def test_embed_texts_does_not_retry_authentication_errors() -> None:
    class UnauthorizedEmbeddings:
        def __init__(self) -> None:
            self.calls = 0

        def create(
            self,
            *,
            model: str,
            input: list[str],
        ) -> SimpleNamespace:
            self.calls += 1
            raise make_authentication_error()

    api = UnauthorizedEmbeddings()
    delays: list[float] = []
    client = EmbeddingClient(
        api,
        model="embedding-model",
        max_attempts=3,
        sleep=delays.append,
    )

    with pytest.raises(AuthenticationError):
        client.embed_texts(["a"])

    assert api.calls == 1
    assert delays == []


def test_embed_texts_rejects_response_count_mismatch() -> None:
    class MissingEmbedding:
        def create(
            self,
            *,
            model: str,
            input: list[str],
        ) -> SimpleNamespace:
            item = SimpleNamespace(
                index=0,
                embedding=[1.0, 2.0],
            )
            return SimpleNamespace(data=[item])

    client = EmbeddingClient(
        MissingEmbedding(),
        model="embedding-model",
        sleep=lambda _: None,
    )

    with pytest.raises(ValueError, match="count"):
        client.embed_texts(["a", "b"])


def test_embed_texts_rejects_duplicate_response_indexes() -> None:
    class DuplicateIndexes:
        def create(
            self,
            *,
            model: str,
            input: list[str],
        ) -> SimpleNamespace:
            data = [
                SimpleNamespace(
                    index=0,
                    embedding=[float(value), 1.0],
                )
                for value in range(len(input))
            ]
            return SimpleNamespace(data=data)

    client = EmbeddingClient(
        DuplicateIndexes(),
        model="embedding-model",
        sleep=lambda _: None,
    )

    with pytest.raises(ValueError, match="indexes"):
        client.embed_texts(["a", "b"])


def test_embed_texts_rejects_dimension_mismatch() -> None:
    class DifferentDimensions:
        def create(
            self,
            *,
            model: str,
            input: list[str],
        ) -> SimpleNamespace:
            data = [
                SimpleNamespace(
                    index=0,
                    embedding=[1.0, 2.0],
                ),
                SimpleNamespace(
                    index=1,
                    embedding=[3.0],
                ),
            ]
            return SimpleNamespace(data=data)

    client = EmbeddingClient(
        DifferentDimensions(),
        model="embedding-model",
        sleep=lambda _: None,
    )

    with pytest.raises(
        ValueError,
        match="^embedding dimension mismatch$",
    ):
        client.embed_texts(["a", "b"])


def test_embed_texts_rejects_zero_dimension_vectors() -> None:
    class EmptyVector:
        def create(
            self,
            *,
            model: str,
            input: list[str],
        ) -> SimpleNamespace:
            item = SimpleNamespace(
                index=0,
                embedding=[],
            )
            return SimpleNamespace(data=[item])

    client = EmbeddingClient(
        EmptyVector(),
        model="embedding-model",
        sleep=lambda _: None,
    )

    with pytest.raises(
        ValueError,
        match="^embedding dimension mismatch$",
    ):
        client.embed_texts(["a"])
