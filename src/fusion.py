from collections import defaultdict
from dataclasses import dataclass

from src.models import SearchResult


@dataclass(frozen=True)
class FusionResult:
    result: SearchResult
    fusion_score: float


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    weights: list[float] | None = None,
    rank_constant: int = 60,
) -> list[FusionResult]:
    if rank_constant <= 0:
        raise ValueError(
            "rank_constant must be positive"
        )

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError(
            "weights must match ranked_lists"
        )

    if any(weight < 0 for weight in weights):
        raise ValueError(
            "fusion weights must be non-negative"
        )

    merged: dict[str, SearchResult] = {}
    scores: dict[str, float] = defaultdict(float)

    for results, weight in zip(ranked_lists, weights):
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.id

            if chunk_id not in merged:
                merged[chunk_id] = result

            scores[chunk_id] += (
                weight / (rank_constant + rank)
            )

    ranked_ids = sorted(
        merged,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    return [
        FusionResult(
            result=merged[chunk_id],
            fusion_score=scores[chunk_id],
        )
        for chunk_id in ranked_ids
    ]