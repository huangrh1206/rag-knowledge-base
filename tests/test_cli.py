from pathlib import Path
from types import SimpleNamespace
import pytest
import numpy as np

from src.cli import build_parser, print_answer
from src.config import Settings
from src.models import Answer, Citation
from src.vector_store import VectorStore
from src.rag_service import IndexReport
from src.index_manifest import IndexManifest, write_manifest
from src.reranker import DisabledReranker
from src.retriever_factory import create_dense_retriever
from pathlib import Path

from src.cli import configure_logging

def test_parser_accepts_index_ask_and_chat_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["index", "data"]).command == "index"
    assert parser.parse_args(["ask", "What is RAG?"]).question == "What is RAG?"
    assert parser.parse_args(["chat"]).command == "chat"


def test_print_answer_includes_citation(capsys) -> None:
    answer = Answer(
        "Use retrieval first [1]",
        (Citation(1, "guide.docx", 2, 3),),
        (),
    )

    print_answer(answer)

    output = capsys.readouterr().out
    assert "Use retrieval first [1]" in output
    assert "[1] guide.docx, paragraphs 2-3" in output


def test_dense_retriever_uses_similarity_threshold(
    tmp_path: Path,
) -> None:
    index_dir = tmp_path / "index"
    VectorStore([], np.empty((0, 2), dtype=np.float32)).save(index_dir)

    manifest = write_manifest(
        index_dir,
        IndexManifest(
            schema_version=1,
            backend="numpy",
            embedding_model="embedding-model",
            vector_dimension=2,
            chunk_size=700,
            chunk_overlap=100,
            document_count=1,
            chunk_count=1,
        ),
    )

    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat-model",
        embedding_model="embedding-model",
        top_k=7,
        similarity_threshold=0.42,
        index_dir=index_dir,
    )
    client = SimpleNamespace(embeddings=object())

    retriever = create_dense_retriever(client, settings)

    assert retriever._threshold == 0.42
    assert retriever._embedder._api is client.embeddings

def test_index_uses_qdrant_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import src.cli as cli_module

    settings = Settings(
        api_key="test-key",
        base_url=None,
        chat_model="chat-model",
        embedding_model="embedding-model",
        vector_store_backend="qdrant",
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="rag_chunks",
    )

    calls: list[dict[str, object]] = []

    class FakeSettings:
        pass

    monkeypatch.setattr(
        cli_module.Settings,
        "from_env",
        classmethod(lambda cls: settings),
    )
    monkeypatch.setattr(
        cli_module,
        "_client",
        lambda current_settings: SimpleNamespace(
            embeddings=object(),
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "_embedder",
        lambda client, current_settings: object(),
    )

    def fake_build_qdrant_index(**kwargs: object) -> IndexReport:
        calls.append(kwargs)
        return IndexReport(
            document_count=1,
            chunk_count=2,
            errors={},
        )

    monkeypatch.setattr(
        cli_module,
        "build_qdrant_index",
        fake_build_qdrant_index,
    )

    monkeypatch.setattr(
        cli_module,
        "build_index",
        lambda **kwargs: pytest.fail(
            "NumPy builder should not be called"
        ),
    )

    monkeypatch.setattr(
        cli_module,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                command="index",
                directory=tmp_path / "documents",
            )
        ),
    )

    assert cli_module.main() == 0
    assert calls[0]["qdrant_path"] == tmp_path / "qdrant"
    assert calls[0]["collection_name"] == "rag_chunks"

def test_configure_logging_creates_dated_log_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    configure_logging()

    log_file = (
        tmp_path
        / "log"
        / f"rag-{__import__('datetime').datetime.now().strftime('%Y-%m-%d')}.log"
    )

    assert log_file.exists()
