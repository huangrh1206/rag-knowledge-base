from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Paragraph:
    text: str
    source: str
    position: int
    is_heading: bool = False

@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    paragraph_start: int
    paragraph_end: int

    def to_dict(self) -> dict[str, object]:
        """递归转换为普通字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Chunk":
        return cls(
            id=str(value["id"]),
            text=str(value["text"]),
            source=str(value["source"]),
            paragraph_start=int(value["paragraph_start"]),
            paragraph_end=int(value["paragraph_end"])
        )
    
@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float

@dataclass(frozen=True)
class Citation:
    number: int
    source: str
    paragraph_start: int
    paragraph_end: int

@dataclass(frozen=True)
class Answer:
    answer: str
    citations: tuple[Citation, ...]
    retrieved_chunks: tuple[SearchResult, ...]