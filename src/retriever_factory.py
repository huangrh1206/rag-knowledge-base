import json
from pathlib import Path

from openai import OpenAI

from src.bm25_retriever import BM25Retriever
from src.config import Settings
from src.embeddings import EmbeddingClient
from src.hybrid_retriever import HybridRetriever
from src.models import Chunk, SearchResult
from src.reranker_factory import reranker
from src.retriever import Retriever
from src.store_factory import load_search_store


def create_embedder(
    client: OpenAI,
    settings: Settings,
) -> EmbeddingClient:
    return EmbeddingClient(
        api=client.embeddings,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )


def create_dense_retriever(
    client: OpenAI,
    settings: Settings,
) -> Retriever:
    store = load_search_store(settings)

    return Retriever(
        embedder=create_embedder(client, settings),
        store=store,
        threshold=settings.similarity_threshold,
    )


def load_chunk_results(
    settings: Settings,
) -> list[SearchResult]:
    if settings.vector_store_backend != "numpy":
        raise ValueError(
            "hybrid retrieval currently requires "
            "the numpy backend because BM25 needs "
            "the complete chunk metadata"
        )

    chunks_path = settings.index_dir / "chunks.json"

    try:
        value = json.loads(
            chunks_path.read_text(
                encoding="utf-8",
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "cannot load chunks for BM25 retrieval"
        ) from exc

    if not isinstance(value, list):
        raise ValueError(
            "chunk metadata must be a list"
        )

    results: list[SearchResult] = []

    for item in value:
        if not isinstance(item, dict):
            raise ValueError(
                "chunk metadata item must be an object"
            )

        chunk = Chunk.from_dict(item)

        results.append(
            SearchResult(
                chunk=chunk,
                score=0.0,
            )
        )

    if not results:
        raise ValueError(
            "cannot build BM25 retrieval from empty chunks"
        )

    return results


def create_bm25_retriever(
    settings: Settings,
) -> BM25Retriever:
    return BM25Retriever(
        results=load_chunk_results(settings),
    )


def create_retriever(
    client: OpenAI,
    settings: Settings,
) -> HybridRetriever:
    dense_retriever = create_dense_retriever(
        client=client,
        settings=settings,
    )

    if not settings.hybrid_enabled:
        return dense_retriever

    bm25_retriever = create_bm25_retriever(
        settings=settings,
    )

    return HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        candidate_k=settings.retrieval_candidate_k,
        top_k=settings.top_k,
        reranker=reranker(settings),
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
        rank_constant=settings.rrf_rank_constant,
    )