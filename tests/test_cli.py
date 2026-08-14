from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.cli import _retriever, build_parser, print_answer
from src.config import Settings
from src.models import Answer, Citation
from src.vector_store import VectorStore


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


def test_retriever_uses_settings_top_k_and_threshold(
    tmp_path: Path,
) -> None:
    index_dir = tmp_path / "index"
    VectorStore([], np.empty((0, 2), dtype=np.float32)).save(index_dir)
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

    retriever = _retriever(client, settings)

    assert retriever._top_k == 7
    assert retriever._threshold == 0.42
    assert retriever._embedder._api is client.embeddings
