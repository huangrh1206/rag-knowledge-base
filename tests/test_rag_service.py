from pathlib import Path

import numpy as np
import pytest
import re
from src.models import Chunk, Paragraph, SearchResult
from src.evidence_policy import EvidencePolicy
from src.generator import INSUFFICIENT_EVIDENCE
from src.rag_service import RAGService, build_index, build_qdrant_index
from src.qdrant_vector_store import QdrantVectorStore
from src.index_manifest import (
    load_manifest,
    qdrant_manifest_directory,
)
from src.retriever import Retriever
from src.hybrid_retriever import HybridRetriever

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
        embedding_model="embedding-model",
        chunk_size=700,
        overlap=100,
    )

    manifest = load_manifest(tmp_path / "index")

    assert report.document_count == 1
    assert report.chunk_count == 1
    assert report.errors == {}
    assert (tmp_path / "index" / "chunks.json").exists()
    assert (tmp_path / "index" / "embeddings.npy").exists()
    assert manifest.backend == "numpy"
    assert manifest.embedding_model == "embedding-model"
    assert manifest.vector_dimension == 2
    assert manifest.chunk_size == 700
    assert manifest.chunk_overlap == 100
    assert manifest.document_count == 1
    assert manifest.chunk_count == 1    

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
            embedding_model="embedding-model",
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
        embedding_model="embedding-model",
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
        embedding_model="embedding-model",
        chunk_size=700,
        overlap=100,
    )

    manifest = load_manifest(
        qdrant_manifest_directory(
            qdrant_path=qdrant_path,
            collection_name="rag_chunks",
        )
    )

    assert report.document_count == 1
    assert report.chunk_count == 1
    assert report.errors == {}
    assert manifest.backend == "qdrant"
    assert manifest.embedding_model == "embedding-model"
    assert manifest.vector_dimension == 2
    assert manifest.chunk_count == 1

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

class TwoResultRetriever:
    def search(self, question: str) -> list[SearchResult]:
        return [
            SearchResult(
                Chunk(
                    "guide-0000",
                    "第一条证据",
                    "first.docx",
                    1,
                    1,
                ),
                0.95,
            ),
            SearchResult(
                Chunk(
                    "guide-0001",
                    "第二条证据",
                    "second.docx",
                    3,
                    4,
                ),
                0.90,
            ),
        ]


class SecondCitationGenerator:
    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        return "答案只使用第二条证据 [2]"


def test_service_maps_only_referenced_citations() -> None:
    service = RAGService(
        retriever=TwoResultRetriever(),
        generator=SecondCitationGenerator(),
    )

    answer = service.ask("测试引用映射")

    assert answer.answer == "答案只使用第二条证据 [2]"
    assert len(answer.retrieved_chunks) == 2
    assert len(answer.citations) == 1
    assert answer.citations[0].number == 2
    assert answer.citations[0].source == "second.docx"
    assert answer.citations[0].paragraph_start == 3
    assert answer.citations[0].paragraph_end == 4

def test_service_logs_retrieval_and_generation(caplog) -> None:
    service = RAGService(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    with caplog.at_level("INFO", logger="src.rag_service"):
        service.ask("How do I declare parameters?")

    messages = [record.getMessage() for record in caplog.records]

    assert any(
        "retrieval completed" in message
        for message in messages
    )
    assert any(
        "generation completed" in message
        for message in messages
    )
    assert any(
        "request completed" in message
        for message in messages
    )

def test_service_uses_one_request_id_for_all_logs(caplog) -> None:
    service = RAGService(
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    with caplog.at_level("INFO", logger="src.rag_service"):
        service.ask("How do I declare parameters?")

    messages = [
        record.getMessage()
        for record in caplog.records
    ]

    request_ids = set()

    for message in messages:
        match = re.search(
            r"request_id=([0-9a-f]{12})",
            message,
        )
        if match:
            request_ids.add(match.group(1))

    assert len(request_ids) == 1

class FailingRetriever:
    def search(self, question: str) -> list[SearchResult]:
        raise RuntimeError("retriever unavailable")


class FailingGenerator:
    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        raise RuntimeError("generator unavailable")

def test_service_logs_retrieval_failure(caplog) -> None:
    service = RAGService(
        retriever=FailingRetriever(),
        generator=FakeGenerator(),
    )

    with caplog.at_level("ERROR", logger="src.rag_service"):
        with pytest.raises(RuntimeError, match="retriever unavailable"):
            service.ask("test")

    assert any(
        "retrieval failed" in record.getMessage()
        for record in caplog.records
    )

class ThreeChunkStore:
    def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                Chunk(
                    id="chunk-a",
                    text="vector result A",
                    source="a.docx",
                    paragraph_start=1,
                    paragraph_end=2,
                ),
                0.95,
            ),
            SearchResult(
                Chunk(
                    id="chunk-b",
                    text="vector result B",
                    source="b.docx",
                    paragraph_start=3,
                    paragraph_end=4,
                ),
                0.90,
            ),
            SearchResult(
                Chunk(
                    id="chunk-c",
                    text="vector result C",
                    source="c.docx",
                    paragraph_start=5,
                    paragraph_end=6,
                ),
                0.85,
            ),
        ]

class ReverseThreeReranker:
    def rerank(
        self,
        question: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        return list(reversed(results))


class FirstAndThirdCitationGenerator:
    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        assert [
            result.chunk.id
            for result in results
        ] == [
            "chunk-c",
            "chunk-b",
            "chunk-a",
        ]

        return "Use the first and third reranked evidence [1] [3]"

class FakeQueryEmbedder:
    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        return np.array(
            [1.0, 0.0],
            dtype=np.float32,
        )

class EmptyBM25Retriever:
    def search(
        self,
        question: str,
        top_k: int,
    ) -> list[SearchResult]:
        return []

def test_reranked_results_keep_correct_citation_mapping() -> None:
    dense_retriever = Retriever(
        embedder=FakeQueryEmbedder(),
        store=ThreeChunkStore(),
        threshold=0.30,
    )

    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=EmptyBM25Retriever(),
        candidate_k=3,
        top_k=3,
        reranker=ReverseThreeReranker(),
        dense_weight=1.0,
        bm25_weight=0.0,
    )

    service = RAGService(
        retriever=retriever,
        generator=FirstAndThirdCitationGenerator(),
    )

    answer = service.ask("test reranked citations")

    assert [
        result.chunk.id
        for result in answer.retrieved_chunks
    ] == [
        "chunk-c",
        "chunk-b",
        "chunk-a",
    ]

    assert [
        citation.number
        for citation in answer.citations
    ] == [1, 3]

    assert [
        citation.source
        for citation in answer.citations
    ] == [
        "c.docx",
        "a.docx",
    ]

    assert answer.citations[0].paragraph_start == 5
    assert answer.citations[0].paragraph_end == 6
    assert answer.citations[1].paragraph_start == 1
    assert answer.citations[1].paragraph_end == 2