from src.bm25_retriever import BM25Retriever
from src.fusion import reciprocal_rank_fusion
from src.models import SearchResult
from src.retriever import Retriever
from src.reranker import DisabledReranker, Reranker

import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Dense + BM25 fusion + optional reranking."""

    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_retriever: BM25Retriever,
        candidate_k: int,
        top_k: int,
        reranker: Reranker | None = None,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rank_constant: int = 60,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be positive"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )

        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError(
                "fusion weights must be non-negative"
            )

        if dense_weight == 0 and bm25_weight == 0:
            raise ValueError(
                "at least one fusion weight must be positive"
            )

        if rank_constant <= 0:
            raise ValueError(
                "rank_constant must be positive"
            )

        self._dense_retriever = dense_retriever
        self._bm25_retriever = bm25_retriever
        self._candidate_k = candidate_k
        self._top_k = top_k
        self._reranker = reranker or DisabledReranker()
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight
        self._rank_constant = rank_constant

    def search(
        self,
        question: str,
    ) -> list[SearchResult]:
        dense_results = []

        if self._dense_weight > 0:
            dense_results = self._dense_retriever.search_candidates(
                question,
                limit=self._candidate_k,
            )
        
        bm25_results = []
        if self._bm25_weight > 0:
            bm25_results = self._bm25_retriever.search(
                question=question,
                top_k=self._candidate_k,
            )

        ranked_lists: list[list[SearchResult]] = []
        weights: list[float] = []

        if self._dense_weight > 0:
            ranked_lists.append(dense_results)
            weights.append(self._dense_weight)

        if self._bm25_weight > 0:
            ranked_lists.append(bm25_results)
            weights.append(self._bm25_weight)
        
        fused = reciprocal_rank_fusion(
            ranked_lists=ranked_lists,
            weights=weights,
            rank_constant=self._rank_constant,
        )

        fused_results = [
            item.result
            for item in fused[: self._candidate_k]
        ]

        try:
            reranked = self._reranker.rerank(
                question=question,
                results=fused_results,
            )

            logger.info(
                "rerank completed reranker=%s "
                "candidate_count=%d returned_count=%d",
                type(self._reranker).__name__,
                len(fused_results),
                min(len(reranked), self._top_k),
            )
        except Exception:
            logger.exception(
                "rerank failed; fallback to vector results "
                "candidate_count=%d",
                len(fused_results),
            )
            reranked = fused_results

        return reranked[: self._top_k]


