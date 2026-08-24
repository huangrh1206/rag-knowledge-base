from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.document_loader import load_directory
from src.models import Answer, Citation, SearchResult, Chunk
from src.text_splitter import split_paragraphs
from src.vector_store import VectorStore
from src.evidence_policy import EvidencePolicy
from src.generator import INSUFFICIENT_EVIDENCE

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
        evidence_policy: EvidencePolicy | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._evidence_policy = evidence_policy or EvidencePolicy(
            minimum_score=0.30,
            minimum_results=1,
        )

    def ask(self, question: str) -> Answer:
        results = self._retriever.search(question)
        decision = self._evidence_policy.evaluate(results)

        if not decision.allowed:
            return Answer(
                answer=INSUFFICIENT_EVIDENCE,
                citations=(),
                retrieved_chunks=tuple(results),
            )
        
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

def format_chunk_for_embedding(chunk: Chunk) -> str:
    document_title = Path(chunk.source).stem

    return (
        f"文档标题：{document_title}\n"
        f"正文：{chunk.text}"
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
        [
            format_chunk_for_embedding(chunk)
            for chunk in chunks
        ]
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
