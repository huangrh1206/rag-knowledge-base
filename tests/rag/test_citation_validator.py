import pytest

from src.rag.citation_validator import validate_citations


def test_validates_existing_citations() -> None:
    result = validate_citations(
        "答案来自资料 [1]，补充说明见 [2]。",
        evidence_count=2,
    )

    assert result.valid is True
    assert result.has_citation is True
    assert result.referenced_numbers == (1, 2)
    assert result.invalid_numbers == ()


def test_rejects_citation_outside_evidence_range() -> None:
    result = validate_citations(
        "答案见 [3]。",
        evidence_count=2,
    )

    assert result.valid is False
    assert result.invalid_numbers == (3,)


def test_rejects_answer_without_citations() -> None:
    result = validate_citations(
        "这是一个没有引用的回答。",
        evidence_count=2,
    )

    assert result.valid is False
    assert result.has_citation is False
    assert result.referenced_numbers == ()


def test_deduplicates_repeated_citations() -> None:
    result = validate_citations(
        "第一处见 [1]，第二处仍见 [1]。",
        evidence_count=1,
    )

    assert result.valid is True
    assert result.referenced_numbers == (1,)


def test_rejects_negative_evidence_count() -> None:
    with pytest.raises(ValueError, match="evidence_count"):
        validate_citations("答案 [1]", evidence_count=-1)
