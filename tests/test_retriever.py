import numpy as np
import pytest

import src.retriever as retriever_module
from src.models import Chunk, SearchResult
from src.retriever import Retriever


class FakeEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeStore:
    def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        chunk = Chunk(
            "guide-0000",
            "Use type annotations",
            "guide.docx",
            4,
            5,
        )
        return [
            SearchResult(chunk, 0.8),
            SearchResult(chunk, 0.2),
        ]


def test_retriever_applies_similarity_threshold() -> None:
    retriever = Retriever(
        FakeEmbedder(),
        FakeStore(),
        top_k=5,
        threshold=0.5,
    )

    results = retriever.search("How do I declare parameters?")

    assert len(results) == 1
    assert results[0].score == 0.8


def test_retriever_rejects_blank_question() -> None:
    retriever = Retriever(
        FakeEmbedder(),
        FakeStore(),
        top_k=5,
        threshold=0.5,
    )

    with pytest.raises(ValueError, match="question cannot be empty"):
        retriever.search("   ")


def test_format_evidence_numbers_source_and_paragraphs() -> None:
    chunk = Chunk(
        "guide-0000",
        "Use type annotations",
        "guide.docx",
        4,
        5,
    )

    text = retriever_module.format_evidence(
        [SearchResult(chunk, 0.8)]
    )

    assert text == (
        "[1] 来源：guide.docx，第 4-5 段\n"
        "内容：Use type annotations"
    )
