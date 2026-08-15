from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.document_loader import load_directory
from src.models import Answer, Citation, SearchResult
from src.text_splitter import split_paragraphs
from src.vector_store import VectorStore


class BatchEmbedder(Protocol):# EmbeddingClient
    # src\embeddings.py
    def embed_texts(
        self,
        texts: list[str],
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class IndexReport:
    document_count: int
    chunk_count: int
    errors: dict[str, str]


class ResultRetriever(Protocol):
    def search(
        self,
        question: str,
    ) -> list[SearchResult]:
        ...


class ResultGenerator(Protocol):
    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        ...


class RAGService:
    def __init__(
        self,
        retriever: ResultRetriever,
        generator: ResultGenerator,
    ) -> None:
        self._retriever = retriever
        self._generator = generator

    def ask(self, question: str) -> Answer:
        results = self._retriever.search(question)

        text = self._generator.generate(
            question,
            results,
        )

        citations = tuple(
            Citation(
                number=number,
                source=result.chunk.source,
                paragraph_start=result.chunk.paragraph_start,
                paragraph_end=result.chunk.paragraph_end,
            )
            for number, result in enumerate(results, start=1)
        )

        return Answer(
            answer=text,
            citations=citations,
            retrieved_chunks=tuple(results),
        )


def build_index(
    document_dir: Path,
    index_dir: Path,
    embedder: BatchEmbedder,
    chunk_size: int,
    overlap: int,
) -> IndexReport:
    documents, errors = load_directory(document_dir)

    chunks = [
        chunk
        for paragraphs in documents.values()
        for chunk in split_paragraphs(
            paragraphs,
            chunk_size,
            overlap,
        )
    ]

    if not chunks:
        raise ValueError("no readable document content found")

    embeddings = embedder.embed_texts(
        [chunk.text for chunk in chunks]
    )

    VectorStore(
        chunks,
        embeddings,
    ).save(index_dir)

    return IndexReport(
        document_count=len(documents),
        chunk_count=len(chunks),
        errors=errors,
    )
