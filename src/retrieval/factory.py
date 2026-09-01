from openai import OpenAI

from src.retrieval.bm25 import BM25Retriever
from src.config import Settings
from src.infrastructure.embeddings import EmbeddingClient
from src.retrieval.hybrid import HybridRetriever
from src.rag.models import SearchResult
from src.retrieval.reranker_factory import reranker
from src.retrieval.dense import Retriever
from src.persistence.factory import load_search_store
from src.persistence.protocol import SearchStore


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
    store: SearchStore | None = None,
) -> Retriever:
    return Retriever(
        embedder=create_embedder(client, settings),
        store=store or load_search_store(settings),
        threshold=settings.similarity_threshold,
    )


def create_bm25_retriever(
    store: SearchStore,
) -> BM25Retriever:
    chunks = store.all_chunks()

    if not chunks:
        raise ValueError(
            "cannot build BM25 retrieval from empty chunks"
        )

    return BM25Retriever(
        results=[
            SearchResult(chunk=chunk, score=0.0)
            for chunk in chunks
        ],
    )


def create_retriever(
    client: OpenAI,
    settings: Settings,
) -> Retriever | HybridRetriever:
    store = load_search_store(settings)
    dense_retriever = create_dense_retriever(
        client=client,
        settings=settings,
        store=store,
    )

    if not settings.hybrid_enabled:
        return dense_retriever

    bm25_retriever = create_bm25_retriever(
        store=store,
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
