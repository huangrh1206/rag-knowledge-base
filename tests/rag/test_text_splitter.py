import pytest

from src.rag.models import Paragraph
from src.rag.text_splitter import _choose_end, _compose, split_paragraphs


def test_short_paragraphs_become_one_chunk() -> None:
    paragraphs = [
        Paragraph("FastAPI 入门", "guide.docx", 3, is_heading=True),
        Paragraph("使用类型注解声明参数。", "guide.docx", 4),
    ]

    chunks = split_paragraphs(
        paragraphs,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 1
    assert chunks[0].id == "guide-0000"
    assert chunks[0].text == "FastAPI 入门\n使用类型注解声明参数。"
    assert chunks[0].source == "guide.docx"
    assert chunks[0].paragraph_start == 3
    assert chunks[0].paragraph_end == 4


def test_long_text_is_split_with_exact_overlap() -> None:
    paragraphs = [
        Paragraph("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "guide.docx", 1),
    ]

    chunks = split_paragraphs(
        paragraphs,
        chunk_size=10,
        overlap=3,
    )

    assert [chunk.text for chunk in chunks] == [
        "ABCDEFGHIJ",
        "HIJKLMNOPQ",
        "OPQRSTUVWX",
        "VWXYZ",
    ]
    assert [chunk.id for chunk in chunks] == [
        "guide-0000",
        "guide-0001",
        "guide-0002",
        "guide-0003",
    ]
    assert all(chunk.paragraph_start == 1 for chunk in chunks)
    assert all(chunk.paragraph_end == 1 for chunk in chunks)


def test_splitter_prefers_nearby_paragraph_boundaries() -> None:
    paragraphs = [
        Paragraph("AAAA", "guide.docx", 10),
        Paragraph("BBBB", "guide.docx", 20),
        Paragraph("CCCC", "guide.docx", 30),
    ]

    chunks = split_paragraphs(
        paragraphs,
        chunk_size=6,
        overlap=1,
    )

    assert [chunk.text for chunk in chunks] == [
        "AAAA",
        "A\nBBBB",
        "B\nCCCC",
    ]
    assert [
        (chunk.paragraph_start, chunk.paragraph_end)
        for chunk in chunks
    ] == [
        (10, 10),
        (10, 20),
        (20, 30),
    ]


def test_boundary_shortening_preserves_forward_progress() -> None:
    paragraphs = [
        Paragraph("AAAA", "guide.docx", 1),
        Paragraph("BBBB", "guide.docx", 2),
    ]
    text, spans = _compose(paragraphs)

    end = _choose_end(
        text,
        spans,
        start=0,
        chunk_size=6,
        overlap=5,
    )

    assert end == 6
    assert end - 5 > 0


def test_separator_window_maps_to_the_following_paragraph() -> None:
    paragraphs = [
        Paragraph("A", "guide.docx", 1),
        Paragraph("B", "guide.docx", 2),
    ]

    chunks = split_paragraphs(
        paragraphs,
        chunk_size=1,
        overlap=0,
    )

    assert [chunk.text for chunk in chunks] == ["A", "\n", "B"]
    assert [
        (chunk.paragraph_start, chunk.paragraph_end)
        for chunk in chunks
    ] == [(1, 1), (2, 2), (2, 2)]


def test_splitter_rejects_overlap_not_smaller_than_chunk() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_paragraphs(
            [],
            chunk_size=10,
            overlap=10,
        )


def test_splitter_rejects_paragraphs_from_different_sources() -> None:
    paragraphs = [
        Paragraph("文档 A", "a.docx", 1),
        Paragraph("文档 B", "b.docx", 1),
    ]

    with pytest.raises(ValueError, match="one source"):
        split_paragraphs(
            paragraphs,
            chunk_size=100,
            overlap=20,
        )


def test_heading_is_kept_with_its_following_paragraph() -> None:
    paragraphs = [
        Paragraph("A" * 10, "guide.docx", 1),
        Paragraph("新章节", "guide.docx", 2, is_heading=True),
        Paragraph("正文内容", "guide.docx", 3),
    ]

    chunks = split_paragraphs(
        paragraphs,
        chunk_size=14,
        overlap=4,
    )

    assert not chunks[0].text.endswith("新章节")
    assert "新章节\n正文内容" in chunks[0].text


def test_consecutive_headings_are_kept_with_the_first_body_paragraph() -> None:
    paragraphs = [
        Paragraph("H" * 12, "guide.docx", 1, is_heading=True),
        Paragraph("子标题", "guide.docx", 2, is_heading=True),
        Paragraph("正文", "guide.docx", 3),
    ]

    chunks = split_paragraphs(
        paragraphs,
        chunk_size=10,
        overlap=2,
    )

    assert chunks[0].text.endswith("子标题\n正文")
    assert chunks[0].paragraph_end == 3
