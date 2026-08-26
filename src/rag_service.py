from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient

import numpy as np

from src.document_loader import load_directory
from src.models import Answer, SearchResult, Chunk
from src.text_splitter import split_paragraphs
from src.vector_store import VectorStore
from src.evidence_policy import EvidencePolicy
from src.generator import INSUFFICIENT_EVIDENCE
from src.citation_validator import validate_citations, citations_for_numbers
from src.qdrant_vector_store import QdrantVectorStore

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

        validation = validate_citations(
            answer=text,
            evidence_count=len(results),
        )

        if not validation.valid:
            invalid_numbers = ", ".join(
                str(number)
                for number in validation.invalid_numbers
            )

            if invalid_numbers:
                raise ValueError(
                    f"model returned invalid citation numbers: {invalid_numbers}"
                )

            raise ValueError(
                "model answer must contain at least one valid citation"
            )

        citations = citations_for_numbers(
            results=results,
            numbers=validation.referenced_numbers,
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

def load_chunks(
    document_dir: Path,
    chunk_size: int,
    overlap: int,
) -> tuple[list[Chunk], dict[str, str], int]:
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

    return chunks, errors, len(documents)

def build_index(
    document_dir: Path,
    index_dir: Path,
    embedder: BatchEmbedder,
    chunk_size: int,
    overlap: int,
) -> IndexReport:
    chunks, errors, documents_count = load_chunks(
        document_dir=document_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )

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
        document_count=documents_count,
        chunk_count=len(chunks),
        errors=errors,
    )


def build_qdrant_index(
    document_dir: Path,
    qdrant_path: Path,
    collection_name: str,
    embedder: BatchEmbedder,
    chunk_size: int,
    overlap: int,
) -> IndexReport:
    chunks, errors, documents_count = load_chunks(
        document_dir=document_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError("no readable document content found")

    embeddings = embedder.embed_texts(
        [
            format_chunk_for_embedding(chunk)
            for chunk in chunks
        ]
    )

    client = QdrantClient(
        path = str(qdrant_path)
    )

    try:
        QdrantVectorStore.create(
            client=client,
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
        )
    finally:
        client.close()


    return IndexReport(
        document_count=documents_count,
        chunk_count=len(chunks),
        errors=errors,
    )