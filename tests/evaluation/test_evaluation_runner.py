from types import SimpleNamespace

from evaluation.run_retrieval import calculate_keyword_coverage
from src.rag.models import Chunk, SearchResult


def result(text: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id="chunk-1",
            text=text,
            source="guide.docx",
            paragraph_start=1,
            paragraph_end=2,
        ),
        score=0.8,
    )


def test_keyword_coverage_returns_one_for_empty_keywords() -> None:
    assert calculate_keyword_coverage([], [result("无关内容")]) == 1.0


def test_keyword_coverage_counts_matched_terms() -> None:
    value = calculate_keyword_coverage(
        ["电话号码", "地址"],
        [result("这里没有电话号码")],
    )

    assert value == 0.5


def test_keyword_coverage_returns_zero_without_match() -> None:
    value = calculate_keyword_coverage(
        ["私人电话"],
        [result("OpenClaw 配置教程")],
    )

    assert value == 0.0
