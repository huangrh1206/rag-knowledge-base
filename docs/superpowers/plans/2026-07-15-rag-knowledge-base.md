# RAG Knowledge Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a command-line RAG knowledge base that indexes `.docx` technical documents, answers with citations, and exposes a handwritten knowledge-search tool to an agent.

**Architecture:** Keep ingestion, retrieval, generation, and interaction behind small Python modules. Persist chunks as JSON and embeddings as NumPy arrays; expose `RAGService.ask()` as the stable boundary used by both the CLI and the agent.

**Tech Stack:** Python 3.11+, `python-docx`, OpenAI-compatible Python client, NumPy, `python-dotenv`, `pytest`

---

## File Map

```text
.env.example                 # Non-secret configuration example
.gitignore                   # Secrets, virtualenv, caches, generated index
requirements.txt             # Runtime and test dependencies
src/__init__.py              # Package marker
src/config.py                # Environment configuration and validation
src/models.py                # Shared immutable data contracts
src/document_loader.py       # Word paragraph extraction
src/text_splitter.py         # Paragraph-aware overlapping chunking
src/embeddings.py            # Embedding API adapter and retry policy
src/vector_store.py          # Atomic persistence and cosine search
src/retriever.py             # Query embedding, thresholding, evidence formatting
src/generator.py             # Grounded prompt and chat completion
src/rag_service.py           # Indexing and question-answering use cases
src/agent.py                 # Bounded tool-calling loop
src/cli.py                   # index, ask, and chat commands
tests/conftest.py            # Reusable sample builders
tests/test_config.py         # Configuration validation
tests/test_document_loader.py# Word extraction tests
tests/test_text_splitter.py  # Chunking tests
tests/test_embeddings.py     # Retry and API adapter tests
tests/test_vector_store.py   # Persistence and similarity tests
tests/test_retriever.py      # Top-K and threshold tests
tests/test_generator.py      # Prompt and grounded-answer tests
tests/test_rag_service.py    # End-to-end service tests with fakes
tests/test_agent.py          # Tool loop tests
tests/test_cli.py            # CLI behavior tests
evaluation/questions.json    # Manual answerable/unanswerable cases
README.md                    # Setup, architecture, demo, limitations, interview notes
```

### Task 1: Project Skeleton, Configuration, And Data Contracts

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/models.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create dependency and repository metadata**

Create `requirements.txt`:

```text
numpy
openai
python-docx
python-dotenv
pytest
```

Create `.gitignore`:

```text
.env
.venv/
__pycache__/
.pytest_cache/
*.pyc
storage/
```

Create `.env.example`:

```dotenv
RAG_API_KEY=your-api-key
RAG_BASE_URL=https://api.openai.com/v1
RAG_CHAT_MODEL=gpt-4.1-mini
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_CHUNK_SIZE=700
RAG_CHUNK_OVERLAP=100
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.30
RAG_INDEX_DIR=storage/index
```

Create an empty `src/__init__.py`.

- [ ] **Step 2: Write failing configuration tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from src.config import Settings


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("RAG_CHAT_MODEL", "chat-model")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("RAG_TOP_K", "3")

    settings = Settings.from_env()

    assert settings.api_key == "test-key"
    assert settings.chat_model == "chat-model"
    assert settings.embedding_model == "embedding-model"
    assert settings.top_k == 3
    assert settings.index_dir == Path("storage/index")


def test_settings_reject_overlap_not_smaller_than_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-key")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")

    with pytest.raises(ValueError, match="overlap"):
        Settings.from_env()
```

- [ ] **Step 3: Run the tests and confirm the missing-module failure**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'src.config'`.

- [ ] **Step 4: Implement configuration and shared models**

Create `src/config.py`:

```python
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str | None
    chat_model: str
    embedding_model: str
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 5
    similarity_threshold: float = 0.30
    index_dir: Path = Path("storage/index")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        api_key = os.getenv("RAG_API_KEY", "").strip()
        if not api_key:
            raise ValueError("RAG_API_KEY is required")
        settings = cls(
            api_key=api_key,
            base_url=os.getenv("RAG_BASE_URL") or None,
            chat_model=os.getenv("RAG_CHAT_MODEL", "gpt-4.1-mini"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "700")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            similarity_threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.30")),
            index_dir=Path(os.getenv("RAG_INDEX_DIR", "storage/index")),
        )
        if settings.chunk_size <= 0:
            raise ValueError("chunk size must be positive")
        if not 0 <= settings.chunk_overlap < settings.chunk_size:
            raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
        if settings.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not -1.0 <= settings.similarity_threshold <= 1.0:
            raise ValueError("similarity threshold must be between -1 and 1")
        return settings
```

Create `src/models.py`:

```python
from dataclasses import asdict, dataclass


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
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Chunk":
        return cls(
            id=str(value["id"]),
            text=str(value["text"]),
            source=str(value["source"]),
            paragraph_start=int(value["paragraph_start"]),
            paragraph_end=int(value["paragraph_end"]),
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
```

- [ ] **Step 5: Run configuration tests**

Run: `python -m pytest tests/test_config.py -v`

Expected: 2 tests PASS.

- [ ] **Step 6: Commit the skeleton**

```powershell
git add requirements.txt .gitignore .env.example src tests/test_config.py
git commit -m "chore: scaffold RAG project"
```

### Task 2: Word Document Loading

**Files:**
- Create: `src/document_loader.py`
- Create: `tests/test_document_loader.py`

- [ ] **Step 1: Write failing Word extraction tests**

Create `tests/test_document_loader.py`:

```python
from pathlib import Path

from docx import Document
import pytest

from src.document_loader import DocumentLoadError, load_docx, load_directory


def make_docx(path: Path) -> None:
    document = Document()
    document.add_heading("FastAPI 入门", level=1)
    document.add_paragraph("  使用   类型注解。  ")
    document.add_paragraph("")
    document.save(path)


def test_load_docx_normalizes_text_and_keeps_positions(tmp_path: Path) -> None:
    path = tmp_path / "guide.docx"
    make_docx(path)

    paragraphs = load_docx(path)

    assert [item.text for item in paragraphs] == ["FastAPI 入门", "使用 类型注解。"]
    assert [item.position for item in paragraphs] == [1, 2]
    assert paragraphs[0].is_heading is True
    assert paragraphs[0].source == "guide.docx"


def test_load_directory_uses_sorted_docx_files(tmp_path: Path) -> None:
    make_docx(tmp_path / "b.docx")
    make_docx(tmp_path / "a.docx")

    documents, errors = load_directory(tmp_path)

    assert list(documents) == ["a.docx", "b.docx"]
    assert errors == {}


def test_load_docx_rejects_empty_document(tmp_path: Path) -> None:
    path = tmp_path / "empty.docx"
    Document().save(path)

    with pytest.raises(DocumentLoadError, match="empty"):
        load_docx(path)
```

- [ ] **Step 2: Confirm the loader tests fail**

Run: `python -m pytest tests/test_document_loader.py -v`

Expected: FAIL during collection because `src.document_loader` does not exist.

- [ ] **Step 3: Implement the Word loader**

Create `src/document_loader.py`:

```python
from collections import OrderedDict
from pathlib import Path
import re

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from src.models import Paragraph


class DocumentLoadError(ValueError):
    pass


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_docx(path: Path) -> list[Paragraph]:
    if path.suffix.lower() != ".docx":
        raise DocumentLoadError(f"unsupported file type: {path.suffix}")
    try:
        document = Document(path)
    except (PackageNotFoundError, ValueError, KeyError) as exc:
        raise DocumentLoadError(f"cannot read {path.name}: {exc}") from exc

    paragraphs: list[Paragraph] = []
    for position, paragraph in enumerate(document.paragraphs, start=1):
        text = _normalize(paragraph.text)
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style else ""
        lowered = style_name.lower()
        paragraphs.append(
            Paragraph(
                text=text,
                source=path.name,
                position=position,
                is_heading=lowered.startswith("heading") or style_name.startswith("标题"),
            )
        )
    if not paragraphs:
        raise DocumentLoadError(f"empty document: {path.name}")
    return paragraphs


def load_directory(path: Path) -> tuple["OrderedDict[str, list[Paragraph]]", dict[str, str]]:
    documents: "OrderedDict[str, list[Paragraph]]" = OrderedDict()
    errors: dict[str, str] = {}
    for file_path in sorted(path.glob("*.docx"), key=lambda item: item.name.lower()):
        try:
            documents[file_path.name] = load_docx(file_path)
        except DocumentLoadError as exc:
            errors[file_path.name] = str(exc)
    return documents, errors
```

- [ ] **Step 4: Run loader tests**

Run: `python -m pytest tests/test_document_loader.py -v`

Expected: 3 tests PASS.

- [ ] **Step 5: Commit Word ingestion**

```powershell
git add src/document_loader.py tests/test_document_loader.py
git commit -m "feat: load Word document paragraphs"
```

### Task 3: Paragraph-Aware Text Splitting

**Files:**
- Create: `src/text_splitter.py`
- Create: `tests/test_text_splitter.py`

- [ ] **Step 1: Write failing splitter tests**

Create `tests/test_text_splitter.py`:

```python
from src.models import Paragraph
from src.text_splitter import split_paragraphs


def test_splitter_keeps_source_ranges_and_overlap() -> None:
    paragraphs = [
        Paragraph("标题", "guide.docx", 1, True),
        Paragraph("A" * 8, "guide.docx", 2),
        Paragraph("B" * 8, "guide.docx", 3),
        Paragraph("C" * 8, "guide.docx", 4),
    ]

    chunks = split_paragraphs(paragraphs, chunk_size=18, overlap=8)

    assert len(chunks) >= 2
    assert chunks[0].source == "guide.docx"
    assert chunks[0].paragraph_start == 1
    assert chunks[0].text.endswith(chunks[1].text[:8])
    assert all(chunk.text for chunk in chunks)


def test_heading_is_not_left_as_the_last_line_of_a_chunk() -> None:
    paragraphs = [
        Paragraph("A" * 10, "guide.docx", 1),
        Paragraph("新章节", "guide.docx", 2, True),
        Paragraph("正文内容", "guide.docx", 3),
    ]

    chunks = split_paragraphs(paragraphs, chunk_size=13, overlap=4)

    assert not chunks[0].text.endswith("新章节")
    assert any("新章节\n正文内容" in chunk.text for chunk in chunks)
```

- [ ] **Step 2: Confirm splitter tests fail**

Run: `python -m pytest tests/test_text_splitter.py -v`

Expected: FAIL during collection because `src.text_splitter` does not exist.

- [ ] **Step 3: Implement deterministic character windows with paragraph mapping**

Create `src/text_splitter.py` with these public and private functions:

```python
from dataclasses import dataclass
from pathlib import Path

from src.models import Chunk, Paragraph


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    position: int
    is_heading: bool


def _compose(paragraphs: list[Paragraph]) -> tuple[str, list[_Span]]:
    parts: list[str] = []
    spans: list[_Span] = []
    cursor = 0
    for index, paragraph in enumerate(paragraphs):
        if index:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(paragraph.text)
        cursor += len(paragraph.text)
        spans.append(_Span(start, cursor, paragraph.position, paragraph.is_heading))
    return "".join(parts), spans


def _paragraph_range(spans: list[_Span], start: int, end: int) -> tuple[int, int]:
    touched = [span.position for span in spans if span.end > start and span.start < end]
    return min(touched), max(touched)


def _choose_end(text: str, spans: list[_Span], start: int, chunk_size: int) -> int:
    proposed = min(start + chunk_size, len(text))
    if proposed == len(text):
        return proposed
    newline = text.rfind("\n", start + chunk_size // 2, proposed + 1)
    end = newline if newline > start else proposed
    ending_span = next((span for span in spans if span.start < end <= span.end), None)
    if ending_span and ending_span.is_heading:
        next_span = next((span for span in spans if span.start > ending_span.start), None)
        if next_span:
            end = next_span.end
    return end


def split_paragraphs(paragraphs: list[Paragraph], chunk_size: int, overlap: int) -> list[Chunk]:
    if not paragraphs:
        return []
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk size")
    sources = {paragraph.source for paragraph in paragraphs}
    if len(sources) != 1:
        raise ValueError("all paragraphs must belong to one source")

    text, spans = _compose(paragraphs)
    source = paragraphs[0].source
    stem = Path(source).stem
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = _choose_end(text, spans, start, chunk_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            paragraph_start, paragraph_end = _paragraph_range(spans, start, end)
            chunks.append(
                Chunk(
                    id=f"{stem}-{len(chunks):04d}",
                    text=chunk_text,
                    source=source,
                    paragraph_start=paragraph_start,
                    paragraph_end=paragraph_end,
                )
            )
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks
```

- [ ] **Step 4: Run splitter tests and adjust only boundary defects exposed by them**

Run: `python -m pytest tests/test_text_splitter.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Run all current tests**

Run: `python -m pytest -v`

Expected: 7 tests PASS.

- [ ] **Step 6: Commit the splitter**

```powershell
git add src/text_splitter.py tests/test_text_splitter.py
git commit -m "feat: split documents into overlapping chunks"
```

### Task 4: Embedding API Adapter And Retry Rules

**Files:**
- Create: `src/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing adapter tests with fake API objects**

Create `tests/test_embeddings.py`:

```python
from types import SimpleNamespace

import numpy as np
import pytest

from src.embeddings import EmbeddingClient


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
        self.calls += 1
        data = [SimpleNamespace(index=index, embedding=[float(index), 1.0]) for index, _ in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))


def test_embed_texts_restores_api_index_order() -> None:
    api = FakeEmbeddings()
    client = EmbeddingClient(api, model="embed-model", batch_size=2, sleep=lambda _: None)

    result = client.embed_texts(["a", "b", "c"])

    np.testing.assert_array_equal(result, np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]))
    assert api.calls == 2


def test_embed_texts_does_not_save_partial_dimension_mismatch() -> None:
    class BadEmbeddings:
        def create(self, *, model: str, input: list[str]) -> SimpleNamespace:
            data = [SimpleNamespace(index=0, embedding=[1.0, 2.0])]
            if len(input) > 1:
                data.append(SimpleNamespace(index=1, embedding=[1.0]))
            return SimpleNamespace(data=data)

    client = EmbeddingClient(BadEmbeddings(), model="embed-model", sleep=lambda _: None)

    with pytest.raises(ValueError, match="dimension"):
        client.embed_texts(["a", "b"])
```

- [ ] **Step 2: Confirm embedding tests fail**

Run: `python -m pytest tests/test_embeddings.py -v`

Expected: FAIL during collection because `src.embeddings` does not exist.

- [ ] **Step 3: Implement batching and deterministic response validation**

Create `src/embeddings.py`:

```python
from collections.abc import Callable, Sequence
import time
from typing import Any

import numpy as np
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


class EmbeddingClient:
    def __init__(
        self,
        api: Any,
        model: str,
        batch_size: int = 64,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api = api
        self._model = model
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._sleep = sleep

    def _request(self, texts: list[str]) -> Any:
        retryable = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._api.create(model=self._model, input=texts)
            except retryable:
                if attempt == self._max_attempts:
                    raise
                self._sleep(2 ** (attempt - 1))
        raise RuntimeError("embedding retry loop ended unexpectedly")

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("at least one text is required")
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self._batch_size):
            batch = list(texts[offset : offset + self._batch_size])
            response = self._request(batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise ValueError("embedding response count mismatch")
            vectors.extend(item.embedding for item in ordered)
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("embedding dimension mismatch")
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
```

- [ ] **Step 4: Run embedding tests**

Run: `python -m pytest tests/test_embeddings.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Commit the embedding adapter**

```powershell
git add src/embeddings.py tests/test_embeddings.py
git commit -m "feat: add embedding API adapter"
```

### Task 5: Persistent NumPy Vector Store

**Files:**
- Create: `src/vector_store.py`
- Create: `tests/test_vector_store.py`

- [ ] **Step 1: Write failing persistence and search tests**

Create `tests/test_vector_store.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from src.models import Chunk
from src.vector_store import IndexFormatError, VectorStore


def chunks() -> list[Chunk]:
    return [
        Chunk("a-0000", "alpha", "a.docx", 1, 1),
        Chunk("a-0001", "beta", "a.docx", 2, 2),
    ]


def test_save_load_and_cosine_search(tmp_path: Path) -> None:
    store = VectorStore(chunks(), np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    results = loaded.search(np.array([0.9, 0.1], dtype=np.float32), top_k=2)

    assert [result.chunk.id for result in results] == ["a-0000", "a-0001"]
    assert results[0].score > results[1].score


def test_load_rejects_chunk_vector_count_mismatch(tmp_path: Path) -> None:
    VectorStore(chunks(), np.ones((2, 2), dtype=np.float32)).save(tmp_path)
    np.save(tmp_path / "embeddings.npy", np.ones((1, 2), dtype=np.float32))

    with pytest.raises(IndexFormatError, match="count"):
        VectorStore.load(tmp_path)
```

- [ ] **Step 2: Confirm vector-store tests fail**

Run: `python -m pytest tests/test_vector_store.py -v`

Expected: FAIL during collection because `src.vector_store` does not exist.

- [ ] **Step 3: Implement validation, atomic save, load, and cosine search**

Create `src/vector_store.py`:

```python
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src.models import Chunk, SearchResult


class IndexFormatError(ValueError):
    pass


class VectorStore:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise IndexFormatError("embeddings must be a two-dimensional array")
        if len(chunks) != embeddings.shape[0]:
            raise IndexFormatError("chunk and embedding count mismatch")
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32, copy=False)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        chunks_target = directory / "chunks.json"
        vectors_target = directory / "embeddings.npy"
        with NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False) as file:
            json.dump([chunk.to_dict() for chunk in self.chunks], file, ensure_ascii=False, indent=2)
            chunks_temp = Path(file.name)
        with NamedTemporaryFile("wb", dir=directory, delete=False) as file:
            np.save(file, self.embeddings)
            vectors_temp = Path(file.name)
        os.replace(chunks_temp, chunks_target)
        os.replace(vectors_temp, vectors_target)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        chunks_path = directory / "chunks.json"
        vectors_path = directory / "embeddings.npy"
        if not chunks_path.exists() or not vectors_path.exists():
            raise IndexFormatError("index files are missing; rebuild the index")
        with chunks_path.open(encoding="utf-8") as file:
            chunks = [Chunk.from_dict(item) for item in json.load(file)]
        embeddings = np.load(vectors_path, allow_pickle=False)
        return cls(chunks, embeddings)

    def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        if query.ndim != 1 or query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("query embedding dimension mismatch")
        matrix_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query)
        denominators = matrix_norms * query_norm
        scores = np.divide(
            self.embeddings @ query,
            denominators,
            out=np.zeros_like(matrix_norms),
            where=denominators != 0,
        )
        indexes = np.argsort(-scores, kind="stable")[: min(top_k, len(self.chunks))]
        return [SearchResult(self.chunks[index], float(scores[index])) for index in indexes]
```

- [ ] **Step 4: Run vector-store tests**

Run: `python -m pytest tests/test_vector_store.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Commit the local index**

```powershell
git add src/vector_store.py tests/test_vector_store.py
git commit -m "feat: persist and search vector index"
```

### Task 6: Retriever And Evidence Formatting

**Files:**
- Create: `src/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write failing retrieval tests**

Create `tests/test_retriever.py`:

```python
import numpy as np

from src.models import Chunk, SearchResult
from src.retriever import Retriever, format_evidence


class FakeEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeStore:
    def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        chunk = Chunk("guide-0000", "类型注解", "guide.docx", 4, 5)
        return [SearchResult(chunk, 0.8), SearchResult(chunk, 0.2)]


def test_retriever_applies_similarity_threshold() -> None:
    retriever = Retriever(FakeEmbedder(), FakeStore(), top_k=5, threshold=0.5)

    results = retriever.search("如何声明参数？")

    assert len(results) == 1
    assert results[0].score == 0.8


def test_format_evidence_numbers_sources() -> None:
    chunk = Chunk("guide-0000", "类型注解", "guide.docx", 4, 5)

    text = format_evidence([SearchResult(chunk, 0.8)])

    assert text == "[1] 来源：guide.docx，第 4-5 段\n内容：类型注解"
```

- [ ] **Step 2: Confirm retriever tests fail**

Run: `python -m pytest tests/test_retriever.py -v`

Expected: FAIL during collection because `src.retriever` does not exist.

- [ ] **Step 3: Implement thresholding and evidence formatting**

Create `src/retriever.py`:

```python
from typing import Protocol

import numpy as np

from src.models import SearchResult


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray:
        pass


class SearchStore(Protocol):
    def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        pass


class Retriever:
    def __init__(self, embedder: QueryEmbedder, store: SearchStore, top_k: int, threshold: float) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._threshold = threshold

    def search(self, question: str) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("question cannot be empty")
        query = self._embedder.embed_query(question)
        return [result for result in self._store.search(query, self._top_k) if result.score >= self._threshold]


def format_evidence(results: list[SearchResult]) -> str:
    blocks: list[str] = []
    for number, result in enumerate(results, start=1):
        chunk = result.chunk
        blocks.append(
            f"[{number}] 来源：{chunk.source}，第 {chunk.paragraph_start}-{chunk.paragraph_end} 段\n"
            f"内容：{chunk.text}"
        )
    return "\n\n".join(blocks)
```

- [ ] **Step 4: Run retriever tests**

Run: `python -m pytest tests/test_retriever.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Commit retrieval orchestration**

```powershell
git add src/retriever.py tests/test_retriever.py
git commit -m "feat: retrieve and format grounded evidence"
```

### Task 7: Grounded Answer Generator

**Files:**
- Create: `src/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write failing generation tests**

Create `tests/test_generator.py`:

```python
from types import SimpleNamespace

from src.generator import INSUFFICIENT_EVIDENCE, AnswerGenerator
from src.models import Chunk, SearchResult


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        message = SimpleNamespace(content="FastAPI 使用类型注解 [1]")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_generator_skips_api_when_evidence_is_empty() -> None:
    api = FakeCompletions()
    generator = AnswerGenerator(api, model="chat-model")

    answer = generator.generate("问题", [])

    assert answer == INSUFFICIENT_EVIDENCE
    assert api.calls == []


def test_generator_sends_numbered_evidence() -> None:
    api = FakeCompletions()
    generator = AnswerGenerator(api, model="chat-model")
    result = SearchResult(Chunk("id", "类型注解", "guide.docx", 2, 2), 0.9)

    answer = generator.generate("如何声明参数？", [result])

    assert answer == "FastAPI 使用类型注解 [1]"
    messages = api.calls[0]["messages"]
    assert "[1] 来源：guide.docx" in messages[1]["content"]
```

- [ ] **Step 2: Confirm generator tests fail**

Run: `python -m pytest tests/test_generator.py -v`

Expected: FAIL during collection because `src.generator` does not exist.

- [ ] **Step 3: Implement the grounded prompt**

Create `src/generator.py`:

```python
from typing import Any

from src.models import SearchResult
from src.retriever import format_evidence


INSUFFICIENT_EVIDENCE = "知识库中没有足够信息回答这个问题。"
SYSTEM_PROMPT = """你是技术知识库助手。
只根据用户消息中提供的资料回答知识库问题。
每个关键结论必须标注资料编号，例如 [1]。
资料不足时回答“知识库中没有足够信息回答这个问题。”，不得补写不存在的事实。"""


class AnswerGenerator:
    def __init__(self, api: Any, model: str) -> None:
        self._api = api
        self._model = model

    def generate(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return INSUFFICIENT_EVIDENCE
        content = f"资料：\n{format_evidence(results)}\n\n问题：{question}"
        response = self._api.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        answer = response.choices[0].message.content
        if not answer:
            raise ValueError("chat model returned empty content")
        return answer.strip()
```

- [ ] **Step 4: Run generator tests**

Run: `python -m pytest tests/test_generator.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Commit grounded generation**

```powershell
git add src/generator.py tests/test_generator.py
git commit -m "feat: generate answers from cited evidence"
```

### Task 8: RAG Service And Indexing Use Case

**Files:**
- Create: `src/rag_service.py`
- Create: `tests/test_rag_service.py`

- [ ] **Step 1: Write failing service tests with fakes**

Create `tests/test_rag_service.py`:

```python
from pathlib import Path

import numpy as np

from src.models import Chunk, SearchResult
from src.rag_service import RAGService, build_index


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


class FakeRetriever:
    def search(self, question: str) -> list[SearchResult]:
        chunk = Chunk("guide-0000", "类型注解", "guide.docx", 1, 2)
        return [SearchResult(chunk, 0.9)]


class FakeGenerator:
    def generate(self, question: str, results: list[SearchResult]) -> str:
        return "使用类型注解 [1]"


def test_service_returns_answer_citations_and_chunks() -> None:
    service = RAGService(FakeRetriever(), FakeGenerator())

    answer = service.ask("如何声明参数？")

    assert answer.answer == "使用类型注解 [1]"
    assert answer.citations[0].source == "guide.docx"
    assert answer.retrieved_chunks[0].score == 0.9


def test_build_index_saves_all_chunks(tmp_path: Path, monkeypatch) -> None:
    from src.models import Paragraph

    monkeypatch.setattr(
        "src.rag_service.load_directory",
        lambda _: ({"guide.docx": [Paragraph("正文", "guide.docx", 1)]}, {}),
    )

    report = build_index(tmp_path / "docs", tmp_path / "index", FakeEmbedder(), 700, 100)

    assert report.document_count == 1
    assert report.chunk_count == 1
    assert (tmp_path / "index" / "chunks.json").exists()
```

- [ ] **Step 2: Confirm service tests fail**

Run: `python -m pytest tests/test_rag_service.py -v`

Expected: FAIL during collection because `src.rag_service` does not exist.

- [ ] **Step 3: Implement indexing and structured answers**

Create `src/rag_service.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.document_loader import load_directory
from src.models import Answer, Citation, SearchResult
from src.text_splitter import split_paragraphs
from src.vector_store import VectorStore


class BatchEmbedder(Protocol):
    def embed_texts(self, texts: list[str]):
        pass


class ResultRetriever(Protocol):
    def search(self, question: str) -> list[SearchResult]:
        pass


class ResultGenerator(Protocol):
    def generate(self, question: str, results: list[SearchResult]) -> str:
        pass


@dataclass(frozen=True)
class IndexReport:
    document_count: int
    chunk_count: int
    errors: dict[str, str]


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
        for chunk in split_paragraphs(paragraphs, chunk_size, overlap)
    ]
    if not chunks:
        raise ValueError("no readable document content found")
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    VectorStore(chunks, embeddings).save(index_dir)
    return IndexReport(len(documents), len(chunks), errors)


class RAGService:
    def __init__(self, retriever: ResultRetriever, generator: ResultGenerator) -> None:
        self._retriever = retriever
        self._generator = generator

    def ask(self, question: str) -> Answer:
        results = self._retriever.search(question)
        text = self._generator.generate(question, results)
        citations = tuple(
            Citation(index, result.chunk.source, result.chunk.paragraph_start, result.chunk.paragraph_end)
            for index, result in enumerate(results, start=1)
        )
        return Answer(text, citations, tuple(results))
```

- [ ] **Step 4: Run service tests**

Run: `python -m pytest tests/test_rag_service.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `python -m pytest -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit the application service**

```powershell
git add src/rag_service.py tests/test_rag_service.py
git commit -m "feat: add RAG indexing and answer service"
```

### Task 9: Bounded Knowledge-Base Agent

**Files:**
- Create: `src/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing tool-loop tests**

Create `tests/test_agent.py` using simple message objects that reproduce the OpenAI-compatible response fields:

```python
import json
from types import SimpleNamespace

import pytest

from src.agent import KnowledgeAgent
from src.models import Chunk, SearchResult


def tool_call(arguments: str = '{"query":"FastAPI 参数","top_k":3}') -> SimpleNamespace:
    function = SimpleNamespace(name="search_knowledge_base", arguments=arguments)
    return SimpleNamespace(id="call-1", function=function)


class FakeCompletions:
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self.messages = messages
        self.calls = 0

    def create(self, **kwargs: object) -> SimpleNamespace:
        message = self.messages[self.calls]
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeRetriever:
    def search(self, question: str) -> list[SearchResult]:
        chunk = Chunk("guide-0000", "类型注解", "guide.docx", 2, 2)
        return [SearchResult(chunk, 0.9)]


def test_agent_executes_search_tool_then_returns_answer() -> None:
    first = SimpleNamespace(content=None, tool_calls=[tool_call()])
    second = SimpleNamespace(content="使用类型注解 [1]", tool_calls=None)
    api = FakeCompletions([first, second])
    agent = KnowledgeAgent(api, "chat-model", FakeRetriever(), max_rounds=3)

    answer = agent.run("FastAPI 如何声明参数？")

    assert answer == "使用类型注解 [1]"
    assert api.calls == 2


def test_agent_rejects_invalid_tool_arguments() -> None:
    first = SimpleNamespace(content=None, tool_calls=[tool_call("not-json")])
    second = SimpleNamespace(content="工具参数无效", tool_calls=None)
    agent = KnowledgeAgent(FakeCompletions([first, second]), "chat-model", FakeRetriever())

    assert agent.run("问题") == "工具参数无效"


def test_agent_stops_at_max_rounds() -> None:
    repeated = SimpleNamespace(content=None, tool_calls=[tool_call()])
    agent = KnowledgeAgent(FakeCompletions([repeated, repeated]), "chat-model", FakeRetriever(), max_rounds=2)

    with pytest.raises(RuntimeError, match="maximum rounds"):
        agent.run("问题")
```

- [ ] **Step 2: Confirm agent tests fail**

Run: `python -m pytest tests/test_agent.py -v`

Expected: FAIL during collection because `src.agent` does not exist.

- [ ] **Step 3: Implement the validated, bounded tool loop**

Create `src/agent.py`:

```python
import json
from typing import Any

from src.retriever import Retriever, format_evidence


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search technical Word documents before answering document-grounded factual questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]

AGENT_PROMPT = """你是技术知识库 Agent。
寒暄和对话控制可以直接回答。
凡是涉及技术文档事实的问题，必须先调用 search_knowledge_base。
最终答案只能使用工具返回的资料，并保留资料编号引用；资料不足时明确说明。"""


class KnowledgeAgent:
    def __init__(self, api: Any, model: str, retriever: Retriever, max_rounds: int = 3) -> None:
        self._api = api
        self._model = model
        self._retriever = retriever
        self._max_rounds = max_rounds

    def _tool_result(self, arguments: str) -> str:
        try:
            value = json.loads(arguments)
            query = value["query"]
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be a non-empty string")
            results = self._retriever.search(query)
            return format_evidence(results) or "知识库中没有足够信息。"
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def run(self, question: str) -> str:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": question},
        ]
        for _ in range(self._max_rounds):
            response = self._api.create(model=self._model, messages=messages, tools=TOOLS, temperature=0)
            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            assistant_message: dict[str, object] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(assistant_message)
            if not tool_calls:
                if not message.content:
                    raise ValueError("agent returned empty content")
                return message.content.strip()
            for call in tool_calls:
                content = (
                    self._tool_result(call.function.arguments)
                    if call.function.name == "search_knowledge_base"
                    else json.dumps({"error": "unknown tool"}, ensure_ascii=False)
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        raise RuntimeError("agent exceeded maximum rounds")
```

- [ ] **Step 4: Run agent tests**

Run: `python -m pytest tests/test_agent.py -v`

Expected: 3 tests PASS.

- [ ] **Step 5: Commit the Agent**

```powershell
git add src/agent.py tests/test_agent.py
git commit -m "feat: add knowledge-base agent loop"
```

### Task 10: Dependency Wiring And CLI

**Files:**
- Create: `src/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI parser and output tests**

Create `tests/test_cli.py`:

```python
from src.cli import build_parser, print_answer
from src.models import Answer, Citation


def test_parser_accepts_index_and_ask_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["index", "data"]).command == "index"
    assert parser.parse_args(["ask", "问题"]).question == "问题"
    assert parser.parse_args(["chat"]).command == "chat"


def test_print_answer_includes_citation(capsys) -> None:
    answer = Answer("回答 [1]", (Citation(1, "guide.docx", 2, 3),), ())

    print_answer(answer)

    output = capsys.readouterr().out
    assert "回答 [1]" in output
    assert "[1] guide.docx，第 2-3 段" in output
```

- [ ] **Step 2: Confirm CLI tests fail**

Run: `python -m pytest tests/test_cli.py -v`

Expected: FAIL during collection because `src.cli` does not exist.

- [ ] **Step 3: Implement parser, dependency factories, and commands**

Create `src/cli.py` with these behaviors:

```python
import argparse
import logging
from pathlib import Path

from openai import OpenAI

from src.agent import KnowledgeAgent
from src.config import Settings
from src.embeddings import EmbeddingClient
from src.generator import AnswerGenerator
from src.models import Answer
from src.rag_service import RAGService, build_index
from src.retriever import Retriever
from src.vector_store import VectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Word document RAG knowledge base")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("index", help="build an index from .docx files")
    index_parser.add_argument("directory", type=Path)
    ask_parser = subparsers.add_parser("ask", help="ask one grounded question")
    ask_parser.add_argument("question")
    subparsers.add_parser("chat", help="start an interactive agent chat")
    return parser


def _client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def _embedder(client: OpenAI, settings: Settings) -> EmbeddingClient:
    return EmbeddingClient(client.embeddings, settings.embedding_model)


def _retriever(client: OpenAI, settings: Settings) -> Retriever:
    store = VectorStore.load(settings.index_dir)
    return Retriever(_embedder(client, settings), store, settings.top_k, settings.similarity_threshold)


def print_answer(answer: Answer) -> None:
    print(answer.answer)
    if answer.citations:
        print("\n来源：")
    for citation in answer.citations:
        print(f"[{citation.number}] {citation.source}，第 {citation.paragraph_start}-{citation.paragraph_end} 段")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    try:
        settings = Settings.from_env()
        client = _client(settings)
        if args.command == "index":
            report = build_index(
                args.directory,
                settings.index_dir,
                _embedder(client, settings),
                settings.chunk_size,
                settings.chunk_overlap,
            )
            print(f"已索引 {report.document_count} 个文档、{report.chunk_count} 个文本块。")
            for name, error in report.errors.items():
                print(f"跳过 {name}: {error}")
            return 0

        retriever = _retriever(client, settings)
        if args.command == "ask":
            service = RAGService(retriever, AnswerGenerator(client.chat.completions, settings.chat_model))
            print_answer(service.ask(args.question))
            return 0

        agent = KnowledgeAgent(client.chat.completions, settings.chat_model, retriever)
        while True:
            question = input("你> ").strip()
            if question.lower() in {"exit", "quit", "退出"}:
                return 0
            if question:
                print(f"助手> {agent.run(question)}")
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run: `python -m pytest tests/test_cli.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Check the CLI help without an API key**

Run: `python -m src.cli --help`

Expected: exit code 0 and help listing `index`, `ask`, and `chat`.

- [ ] **Step 6: Commit the CLI**

```powershell
git add src/cli.py tests/test_cli.py
git commit -m "feat: add RAG command-line interface"
```

### Task 11: Evaluation Data, Documentation, And Final Verification

**Files:**
- Create: `evaluation/questions.json`
- Create: `README.md`
- Modify: tests only when final verification exposes a documented defect

- [ ] **Step 1: Add a concrete manual evaluation schema**

Create `evaluation/questions.json`:

```json
[
  {
    "question": "FastAPI 如何声明请求参数？",
    "category": "answerable",
    "expected_source": "sample-fastapi-guide.docx"
  },
  {
    "question": "文档推荐使用哪个数据库？",
    "category": "ambiguous",
    "expected_source": null
  },
  {
    "question": "明天上海的天气如何？",
    "category": "unanswerable",
    "expected_source": null
  }
]
```

Expand this file to 10 to 20 questions only after the user supplies the actual Word documents, because expected sources must match real file names.

- [ ] **Step 2: Write the README with runnable commands and interview notes**

Create `README.md` with these sections and exact commands:

```markdown
# Word RAG Knowledge Base

A handwritten learning project that parses `.docx` files, stores embeddings locally, retrieves evidence with cosine similarity, and generates cited answers.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`, place `.docx` files in `data/`, then run:

```powershell
python -m src.cli index .\data
python -m src.cli ask "FastAPI 如何声明请求参数？"
python -m src.cli chat
```

## Architecture

`.docx -> paragraphs -> chunks -> embeddings -> NumPy Top-K -> grounded prompt -> cited answer`

## Data Files

- `chunks.json` stores chunk text and Word paragraph ranges.
- `embeddings.npy` stores vectors in the same order.
- A count mismatch makes the index invalid and requires rebuilding.

## Agent

The agent may answer greetings directly. Technical document questions require the `search_knowledge_base` tool. The local loop validates arguments and stops after three rounds.

## Testing

```powershell
python -m pytest -v
```

Tests use fake API responses and do not spend API credit.

## Limitations

- `.docx` paragraphs only; no legacy `.doc`, OCR, images, or table extraction.
- NumPy full-scan retrieval targets learning-sized indexes.
- Similarity threshold tuning depends on the embedding model and evaluation set.

## Interview Discussion

- RAG adds external, updateable evidence and citations to an LLM call.
- Chunk size trades local precision against contextual completeness.
- Overlap reduces boundary information loss but increases index size and duplicate retrieval.
- Retrieval relevance and answer faithfulness must be evaluated separately.
- FAISS, Milvus, or Elasticsearch becomes appropriate when scale, filtering, hybrid search, or operations require it.
```

- [ ] **Step 3: Run automated verification**

Run: `python -m pytest -v`

Expected: all tests PASS with no live network calls.

- [ ] **Step 4: Run static syntax verification**

Run: `python -m compileall -q src tests`

Expected: exit code 0 with no output.

- [ ] **Step 5: Perform the live smoke test with a configured API**

Place one real `.docx` technical document in `data/`, configure `.env`, then run:

```powershell
python -m src.cli index .\data
python -m src.cli ask "请总结文档的核心内容。"
```

Expected: indexing reports at least 1 document and 1 chunk; the answer includes at least one `[1]` marker and prints a matching file name and paragraph range.

- [ ] **Step 6: Run an unanswerable-question smoke test**

Run: `python -m src.cli ask "文档没有提到的虚构产品价格是多少？"`

Expected: the answer states that the knowledge base has insufficient information and does not invent a price.

- [ ] **Step 7: Inspect repository state and commit documentation**

Run: `git status --short`

Expected: only `README.md` and `evaluation/questions.json` are uncommitted.

```powershell
git add README.md evaluation/questions.json
git commit -m "docs: add setup and RAG evaluation guide"
```

## Plan Self-Review Notes

- Every included requirement in the approved design maps to Tasks 1 through 11.
- FastAPI, browser UI, reranking, hybrid retrieval, and production vector databases remain explicitly outside this plan.
- Shared field names remain `answer`, `citations`, and `retrieved_chunks` from `src.models.Answer` through the service and CLI.
- All paid API behavior is behind fakes in automated tests; only the final smoke test requires network access and credentials.
- The implementation first confirms the selected provider supports OpenAI-compatible embeddings, chat completions, and tool calling.
