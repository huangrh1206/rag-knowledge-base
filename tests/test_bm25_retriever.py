from src.bm25_retriever import BM25Retriever
from src.models import Chunk, SearchResult
import pytest

def make_result(
    chunk_id: str,
    text: str,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            text=text,
            source="guide.docx",
            paragraph_start=1,
            paragraph_end=1,
        ),
        score=0.0,
    )


def test_bm25_returns_keyword_relevant_chunk_first() -> None:
    results = [
        make_result(
            "chunk-a",
            "OpenClaw 可以配置多个智能体。",
        ),
        make_result(
            "chunk-b",
            "可以通过 cron 配置周期性定时任务。",
        ),
        make_result(
            "chunk-c",
            "Ubuntu 支持远程图形桌面连接。",
        ),
    ]

    retriever = BM25Retriever(results)

    returned = retriever.search(
        question="如何配置 cron 定时任务",
        top_k=2,
    )

    assert returned[0].chunk.id == "chunk-b"

def test_bm25_rejects_blank_question() -> None:
    retriever = BM25Retriever(
        [
            make_result("chunk-a", "some text"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        retriever.search("", top_k=1)

def test_bm25_rejects_non_positive_top_k() -> None:
    retriever = BM25Retriever(
        [
            make_result("chunk-a", "some text"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="top_k must be positive",
    ):
        retriever.search("text", top_k=0)