import re
from dataclasses import dataclass

from src.models import Citation, SearchResult

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

@dataclass(frozen=True)
class CitationValidation:
    valid: bool
    referenced_numbers: tuple[int, ...]
    invalid_numbers: tuple[int, ...]
    has_citation: bool


def validate_citations(
    answer: str,
    evidence_count: int,
) -> CitationValidation:
    if evidence_count < 0:
        raise ValueError("evidence_count must be non-negative")

    numbers = tuple(
        int(value)
        for value in _CITATION_PATTERN.findall(answer)
    )

    unique_numbers = tuple(dict.fromkeys(numbers))

    invalid_numbers = tuple(
        number
        for number in unique_numbers
        if number < 1 or number > evidence_count
    )

    return CitationValidation(
        valid=bool(numbers) and not invalid_numbers,
        referenced_numbers=unique_numbers,
        invalid_numbers=invalid_numbers,
        has_citation=bool(numbers),
    )

def citations_for_numbers(
    results: list[SearchResult],
    numbers:tuple[int, ...],
) -> tuple[Citation, ...]:
    return tuple(
        Citation(
            number=number,
            source=results[number - 1].chunk.source,
            paragraph_start=results[number - 1].chunk.paragraph_start,
            paragraph_end=results[number - 1].chunk.paragraph_end,
        )
        for number in numbers
    )