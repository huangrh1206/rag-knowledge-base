from src.models import Answer, Chunk, Citation, Paragraph, SearchResult


def test_paragraph_marks_heading_with_is_heading() -> None:
    paragraph = Paragraph(
        text="FastAPI 入门",
        source="guide.docx",
        position=1,
        is_heading=True,
    )

    assert paragraph.is_heading is True


def test_chunk_round_trips_through_dict() -> None:
    chunk = Chunk(
        id="guide-0001",
        text="FastAPI 使用类型注解。",
        source="guide.docx",
        paragraph_start=2,
        paragraph_end=3,
    )

    restored = Chunk.from_dict(chunk.to_dict())

    assert restored == chunk


def test_answer_keeps_citations_and_retrieval_results() -> None:
    chunk = Chunk("guide-0001", "正文", "guide.docx", 2, 3)
    result = SearchResult(chunk=chunk, score=0.91)
    citation = Citation(
        number=1,
        source="guide.docx",
        paragraph_start=2,
        paragraph_end=3,
    )

    answer = Answer(
        answer="回答 [1]",
        citations=(citation,),
        retrieved_chunks=(result,),
    )

    assert answer.citations[0].source == "guide.docx"
    assert answer.retrieved_chunks[0].score == 0.91
