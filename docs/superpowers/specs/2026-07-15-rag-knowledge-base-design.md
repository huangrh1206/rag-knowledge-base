# RAG Knowledge Base Design

## 1. Goal

Build a small RAG knowledge base in one week for learning and internship interview demonstrations. The implementation must keep the core mechanics visible: Word parsing, chunking, embedding, cosine similarity, Top-K retrieval, prompt assembly, citations, and an agent tool loop.

The first version is a command-line application. Its core service interface must remain independent from the CLI so a FastAPI endpoint and web chat can be added later without rewriting retrieval or generation.

## 2. Scope

### Included

- Read `.docx` technical documents from a local directory.
- Clean paragraphs and split them into overlapping chunks.
- Generate embeddings through an OpenAI-compatible API.
- Persist chunk metadata and embeddings locally.
- Retrieve relevant chunks with a handwritten NumPy cosine-similarity search.
- Generate grounded answers with source citations.
- Refuse knowledge-base answers when retrieved evidence is insufficient.
- Provide single-question and interactive-chat CLI commands.
- Implement a small agent loop with a knowledge-base search tool.
- Add focused unit tests and a small manual evaluation set.
- Expose a stable `rag_service.ask()` interface for a future API layer.

### Excluded From The First Version

- Legacy `.doc` parsing.
- Web UI and FastAPI routes.
- Authentication, authorization, and multi-user isolation.
- Distributed vector databases.
- OCR, images, tables, and advanced Word layout extraction.
- Hybrid search, reranking, and production observability.

## 3. Technical Approach

Use a handwritten RAG pipeline with a small dependency set:

- Python for the application.
- `python-docx` for `.docx` paragraph extraction.
- An OpenAI-compatible client for chat completion, tool calling, and embeddings.
- NumPy for vector persistence and similarity calculations.
- JSON for chunk metadata.
- `pytest` for automated tests.

LangChain and LlamaIndex are intentionally excluded from the first version. They are useful future comparison points, but their abstractions would hide the mechanics this project is intended to teach.

## 4. Architecture

```text
.docx files
    |
    v
document loader -> text cleaner -> chunker
    |
    v
embedding API -> chunks.json + embeddings.npy
    |
    v
question -> question embedding -> cosine similarity -> Top-K chunks
    |
    v
prompt assembly -> chat completion -> cited answer
```

Proposed modules:

```text
src/
  config.py
  document_loader.py
  text_splitter.py
  embeddings.py
  vector_store.py
  retriever.py
  generator.py
  rag_service.py
  agent.py
  cli.py
```

Responsibilities:

- `config.py`: read environment variables and validate model and retrieval settings.
- `document_loader.py`: extract non-empty `.docx` paragraphs and their original positions.
- `text_splitter.py`: create overlapping chunks while retaining source metadata.
- `embeddings.py`: batch embedding requests and retry transient failures.
- `vector_store.py`: atomically save/load index files and run cosine-similarity Top-K search.
- `retriever.py`: embed queries, apply score thresholds, and return cited evidence.
- `generator.py`: build grounded prompts and generate final answers.
- `rag_service.py`: expose indexing and `ask(question)` application operations.
- `agent.py`: define tools and execute the bounded tool-calling loop.
- `cli.py`: provide index, ask, and chat commands.

## 5. Document And Index Model

Each chunk has the following logical shape:

```json
{
  "id": "python-guide-0007",
  "text": "FastAPI uses type annotations...",
  "source": "Python-guide.docx",
  "paragraph_start": 18,
  "paragraph_end": 21
}
```

The loader removes empty paragraphs and normalizes repeated whitespace. The splitter accumulates adjacent paragraphs toward a default target of 700 characters. Neighboring chunks retain 100 characters of overlap by default. Both values are configurable. A paragraph is treated as a heading when its Word style name starts with `Heading` or the localized equivalent `标题`; the splitter keeps that heading with the first following content paragraph.

`chunks.json` stores ordered metadata and text. `embeddings.npy` stores a two-dimensional float array in the same order. Loading fails when the number of chunk records and embedding rows differs. Index creation writes temporary files first and replaces the active files only after the complete embedding batch succeeds, so partial indexes are not exposed.

The index command rebuilds the index explicitly. Query and chat commands load the existing index and never generate document embeddings automatically.

## 6. Retrieval And Answer Generation

For each question:

1. Request its embedding.
2. Normalize vectors and calculate cosine similarity with NumPy.
3. Select a configurable Top-K, initially 5.
4. Remove results below a configurable similarity threshold.
5. Format remaining chunks as numbered evidence blocks.
6. Ask the chat model to answer using only that evidence.

Evidence is formatted with stable citation numbers:

```text
[1] Source: Python-guide.docx, paragraphs 18-21
Content: ...
```

The system prompt requires a citation such as `[1]` for every material claim. When no chunk passes the threshold, the application returns a deterministic insufficient-evidence response without asking the chat model to invent an answer. When evidence exists but does not answer the question, the prompt requires the model to state that the knowledge base lacks sufficient information.

The initial threshold is a configuration value, not a claimed universal optimum. It will be selected using the evaluation questions and documented with the chosen embedding model because similarity scores are model-dependent.

## 7. Agent Design

The agent exposes one external tool in the first version:

- `search_knowledge_base(query, top_k)`: returns numbered chunks, similarity scores, file names, and paragraph ranges.

The model may answer greetings or conversational control messages directly. It must call the search tool before answering questions that assert facts from the technical knowledge base.

The application owns the agent loop:

1. Send the conversation, tool schema, and agent instructions to the model.
2. Inspect any tool-call request.
3. Validate arguments and execute the local search tool.
4. Append the tool result to the conversation.
5. Request the final response.
6. Stop after at most three model/tool rounds.

The first version requires a provider that implements OpenAI-compatible tool calling. A strictly validated JSON-routing adapter is a possible later extension, but it is not part of this scope and the implementation will not maintain two agent paths.

## 8. Configuration And Errors

Configuration is supplied by environment variables and documented in `.env.example`. It includes the API base URL, API key, chat model, embedding model, chunk target, overlap, Top-K, similarity threshold, and index directory. Secrets are never committed.

Error behavior:

- Unsupported `.doc` files, corrupt `.docx` files, and empty documents produce clear per-file errors.
- Transient API timeouts, rate limits, and server errors use exponential backoff with at most three retries.
- Authentication and invalid-request failures are not retried.
- A failed embedding batch leaves the previous complete index intact.
- Missing or inconsistent index files prevent querying and instruct the user to rebuild.
- Invalid agent tool arguments return a structured tool error; the loop remains bounded.
- The CLI prints concise user-facing errors while detailed diagnostics go through Python logging.

## 9. Testing And Evaluation

Automated tests cover:

- Paragraph extraction and source positions.
- Chunk size, overlap, ordering, and preservation of text.
- Cosine-similarity calculations and Top-K ordering.
- Index persistence and mismatch detection.
- Threshold-based insufficient-evidence behavior.
- Prompt evidence and citation formatting.
- Agent tool execution and maximum-round termination.
- API retry classification using fake client responses, without paid API calls.

A manual evaluation set contains 10 to 20 questions divided into answerable, ambiguous, and unanswerable groups. For each question, record whether a relevant chunk appears in Top-K, whether the answer is supported, and whether its citations are correct. This is a learning-oriented baseline rather than a production RAG evaluation framework.

## 10. CLI And Extension Boundary

The target commands are:

```powershell
python -m src.cli index .\data
python -m src.cli ask "How does FastAPI declare request parameters?"
python -m src.cli chat
```

The CLI calls application services and contains no retrieval or generation logic. `rag_service.ask(question)` returns a structured result with `answer`, `citations`, and `retrieved_chunks` fields. Each citation contains its source file and paragraph range; retrieved chunks additionally contain text and score. A future FastAPI route can serialize this result directly, and a web chat can consume that route.

## 11. Acceptance Criteria

- Multiple `.docx` files can be indexed from one directory.
- A saved index loads across process restarts without regenerating document embeddings.
- Answers cite file names and paragraph ranges.
- Unanswerable questions receive an explicit insufficient-evidence response.
- The agent calls the knowledge-base tool for document-grounded factual questions.
- Core logic passes focused automated tests without live API spending.
- API and retrieval settings are configurable.
- The core RAG service has no dependency on the CLI.
- The README explains architecture, setup, commands, an example session, limitations, and interview talking points.

## 12. One-Week Learning Sequence

1. Day 1: learn the RAG flow; implement Word extraction and cleaning.
2. Day 2: implement chunking and embedding calls; learn vectors and cosine similarity.
3. Day 3: implement local persistence and Top-K retrieval.
4. Day 4: implement prompt assembly, cited generation, and insufficient-evidence handling.
5. Day 5: implement the agent loop and CLI question/chat flows.
6. Day 6: add tests, retries, configuration, logging, and the evaluation set.
7. Day 7: finish documentation, prepare a demonstration, and practice explaining design tradeoffs.

## 13. Interview Talking Points

The finished project must support clear explanations of:

- What RAG adds beyond a direct LLM call.
- How chunk size and overlap affect recall and context quality.
- The roles of embeddings, cosine similarity, and Top-K.
- Why relevant retrieval can still produce an incorrect answer.
- How citations and insufficient-evidence behavior reduce hallucination risk.
- How an agent differs from a fixed RAG pipeline.
- When to replace NumPy search with FAISS, Milvus, or Elasticsearch.
- How retrieval recall and answer faithfulness can be evaluated separately.

## 14. Future Extensions

After the learning version is stable, extensions may be added in this order:

1. FastAPI endpoint over `rag_service.ask()`.
2. Minimal browser chat client.
3. Reranking and query rewriting.
4. Hybrid keyword and vector retrieval.
5. A production vector database and operational monitoring.
