from dataclasses import dataclass
from pathlib import Path

from src.models import Chunk, Paragraph


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    position: int
    is_heading: bool


def _compose(
    paragraphs: list[Paragraph],
) -> tuple[str, list[_Span]]:
    parts: list[str] = []
    spans: list[_Span] = []
    cursor = 0

    for index, paragraph in enumerate(paragraphs):
        start = cursor

        if index > 0:
            parts.append("\n")
            cursor += 1

        parts.append(paragraph.text)
        cursor += len(paragraph.text)

        spans.append(
            _Span(
                start=start,
                end=cursor,
                position=paragraph.position,
                is_heading=paragraph.is_heading,
            )
        )

    return "".join(parts), spans


def _choose_end(
        text: str,
        spans: list[_Span],
        start: int,
        chunk_size: int,
        overlap: int,
) -> int:

    proposed = min(
        start + chunk_size,
        len(text)
    )

    if proposed == len(text):
        return proposed

    search_start = max(
        start + chunk_size // 2,
        start + overlap + 1
    )

    newline = text.rfind(
        "\n",
        search_start,
        proposed + 1,
    )

    end = (
        newline
        if newline > start + overlap
        else proposed
    )

    for index, span in enumerate(spans):
        if span.start < end <= span.end:
            if span.is_heading:
                content_index = index + 1

                while (
                    content_index < len(spans)
                    and spans[content_index].is_heading
                ):
                    content_index += 1

                if content_index < len(spans):
                    return spans[content_index].end
            break
    return end


def _paragraph_range(
        spans: list[_Span],
    start: int,
    end: int,
) -> tuple[int, int]:
    """Return the Word paragraph positions touched by a character window."""
    touched = [
        span.position
        for span in spans
        if span.end > start
        and span.start < end
    ]
    return min(touched), max(touched)


def split_paragraphs(
    paragraphs: list[Paragraph],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    if overlap >= chunk_size or overlap < 0:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk size"
        )

    if not paragraphs:
        return []

    sources = {paragraph.source for paragraph in paragraphs}

    if len(sources) != 1:
        raise ValueError(
            "all paragraphs must belong to one source"
        )

    source = paragraphs[0].source

    text, spans = _compose(paragraphs)

    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = _choose_end(
            text,
            spans,
            start,
            chunk_size,
            overlap
        )

        paragraph_start, paragraph_end = _paragraph_range(
            spans,
            start,
            end,
        )

        chunks.append(
            Chunk(
                id=f"{Path(source).stem}-{len(chunks):04d}",
                text=text[start:end],
                source=source,
                paragraph_start=paragraph_start,
                paragraph_end=paragraph_end,
            )
        )

        if end == len(text):
            break

        start = end - overlap

    return chunks

