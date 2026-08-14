# Word RAG Knowledge Base

This is a handwritten learning project that loads `.docx` files, splits them
into paragraph-aware chunks, stores embeddings locally, retrieves evidence with
cosine similarity, and generates answers with citations.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` with an OpenAI-compatible API key. Put Word documents in `data/`:

```powershell
python -m src.cli index .\data
python -m src.cli ask "How does the document define request parameters?"
python -m src.cli chat
```

## Architecture

```text
.docx -> paragraphs -> chunks -> embeddings -> NumPy Top-K
      -> grounded prompt -> cited answer
```

The stable application boundary is `RAGService.ask()`. A future FastAPI route
or browser chat can call the same service without moving retrieval logic into
the frontend.

## Stored index

- `chunks.json` stores chunk text and Word paragraph ranges.
- `embeddings.npy` stores vectors in the same order as the chunks.
- A count or dimension mismatch invalidates the index and requires rebuilding.

## Agent

The agent can answer greetings directly. Technical document questions must use
the `search_knowledge_base` tool. The local loop validates tool arguments and
stops after a bounded number of rounds.

## Testing

```powershell
python -m pytest -q --basetemp=E:\code\RAG\pytest-user-temp
python -m compileall -q src tests
```

Tests use fake API responses and do not spend API credits.

## Limitations

- Only `.docx` paragraphs are loaded; legacy `.doc`, OCR, images, and tables
  are outside this first version.
- NumPy full-scan retrieval is intended for learning-sized indexes.
- Similarity thresholds should be tuned against real evaluation questions.

## Interview points

- RAG supplies updateable external evidence and citations to an LLM.
- Smaller chunks improve local precision; larger chunks preserve context.
- Overlap reduces boundary information loss but increases index size.
- Retrieval relevance and answer faithfulness are separate evaluation targets.
- FAISS, Milvus, or Elasticsearch become useful when scale, filtering, hybrid
  search, or operational requirements exceed this learning implementation.
