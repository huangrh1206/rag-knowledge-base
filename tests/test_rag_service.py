from pathlib import Path

import numpy as np
import pytest

from src.models import Chunk, Paragraph, SearchResult
from src.rag_service import RAGService, build_index
from src.evidence_policy import EvidencePolicy
from src.generator import INSUFFICIENT_EVIDENCE
from src.rag_service import RAGService, build_index, build_qdrant_index
from src.qdrant_vector_store import QdrantVectorStore

class FakeRetriever:
    def search(self, question: str) -> list[SearchResult]:
        chunk = Chunk(
            "guide-0000",
            "Use type annotations",
            "guide.docx",
            1,
            2,
        )
        return [SearchResult(chunk, 0.9)]


class FakeGenerator:
    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        return "Use type annotations [1]"


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


def test_service_returns_answer_citations_and_chunks() -> None:
    service = RAGService(
        FakeRetriever(),
        FakeGenerator(),
    )

    answer = service.ask("How do I declare parameters?")

    assert answer.answer == "Use type annotations [1]"
    assert len(answer.citations) == 1
    assert answer.citations[0].number == 1
    assert answer.citations[0].source == "guide.docx"
    assert answer.citations[0].paragraph_start == 1
    assert answer.citations[0].paragraph_end == 2
    assert answer.retrieved_chunks[0].score == 0.9


def test_build_index_saves_all_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.rag_service.load_directory",
        lambda _: (
            {
                "guide.docx": [
                    Paragraph("Document body", "guide.docx", 1)
                ]
            },
            {},
        ),
    )

    report = build_index(
        document_dir=tmp_path / "docs",
        index_dir=tmp_path / "index",
        embedder=FakeEmbedder(),
        chunk_size=700,
        overlap=100,
    )

    assert report.document_count == 1
    assert report.chunk_count == 1
    assert report.errors == {}
    assert (tmp_path / "index" / "chunks.json").exists()
    assert (tmp_path / "index" / "embeddings.npy").exists()


def test_build_index_rejects_documents_without_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.rag_service.load_directory",
        lambda _: ({}, {"broken.docx": "empty document"}),
    )

    with pytest.raises(ValueError, match="no readable document content"):
        build_index(
            document_dir=tmp_path / "docs",
            index_dir=tmp_path / "index",
            embedder=FakeEmbedder(),
            chunk_size=700,
            overlap=100,
        )

    assert not (tmp_path / "index").exists()

class EmptyRetriever:
    def search(self, question: str) -> list[SearchResult]:
        return []

class RecordingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        self.calls += 1
        return "should not be called"

def test_service_refuses_without_sufficient_evidence() -> None:
    generator = RecordingGenerator()
    service = RAGService(
        retriever=EmptyRetriever(),
        generator=generator,
        evidence_policy=EvidencePolicy(
            minimum_score=0.60,
            minimum_results=1,
        ),
    )

    answer = service.ask("What is the private phone number?")

    assert answer.answer == INSUFFICIENT_EVIDENCE
    assert answer.citations == ()
    assert answer.retrieved_chunks == ()
    assert generator.calls == 0

class RecordingEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.texts = texts
        return np.ones((len(texts), 2), dtype=np.float32)

def test_build_index_enriches_embedding_text_with_document_title(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.rag_service.load_directory",
        lambda _: (
            {
                "openclaw通过mcp调用浏览器.docx": [
                    Paragraph(
                        text="安装 Playwright 并配置服务",
                        source="openclaw通过mcp调用浏览器.docx",
                        position=1,
                    )
                ]
            },
            {},
        ),
    )
    embedder = RecordingEmbedder()

    build_index(
        document_dir=tmp_path / "documents",
        index_dir=tmp_path / "index",
        embedder=embedder,
        chunk_size=700,
        overlap=100,
    )

    assert len(embedder.texts) == 1
    assert "文档标题：openclaw通过mcp调用浏览器" in embedder.texts[0]
    assert "正文：安装 Playwright 并配置服务" in embedder.texts[0]

class InvalidCitationGenerator:
    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        return "答案引用了不存在的资料 [99]"


class NoCitationGenerator:
    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        return "这是一个没有引用的答案"

def test_service_rejects_invalid_citation_number() -> None:
    service = RAGService(
        retriever=FakeRetriever(),
        generator=InvalidCitationGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="invalid citation numbers",
    ):
        service.ask("How do I declare parameters?")

def test_service_rejects_answer_without_citation() -> None:
    service = RAGService(
        retriever=FakeRetriever(),
        generator=NoCitationGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="at least one valid citation",
    ):
        service.ask("How do I declare parameters?")

def test_build_qdrant_index_writes_collection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.rag_service.load_directory",
        lambda _: (
            {
                "guide.docx": [
                    Paragraph(
                        "Document body",
                        "guide.docx",
                        1,
                    )
                ]
            },
            {},
        ),
    )

    qdrant_path = tmp_path / "qdrant"
    embedder = FakeEmbedder()

    report = build_qdrant_index(
        document_dir=tmp_path / "documents",
        qdrant_path=qdrant_path,
        collection_name="rag_chunks",
        embedder=embedder,
        chunk_size=700,
        overlap=100,
    )

    assert report.document_count == 1
    assert report.chunk_count == 1
    assert report.errors == {}

    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(qdrant_path))

    try:
        store = QdrantVectorStore.load(
            client=client,
            collection_name="rag_chunks",
        )

        results = store.search(
            np.array([1.0, 1.0], dtype=np.float32),
            top_k=1,
        )

        assert len(results) == 1
        assert results[0].chunk.source == "guide.docx"
    finally:
        client.close()