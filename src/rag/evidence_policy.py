from src.rag.models import EvidenceDecision, SearchResult

class EvidencePolicy:
    def __init__(
        self,
        minimum_score: float,
        minimum_results: int = 1,
    ) -> None:
        self._minimum_score = minimum_score
        self._minimum_results = minimum_results

    def evaluate(
        self,
        results: list[SearchResult],
    ) -> EvidenceDecision:
        if len(results) < self._minimum_results:
            return EvidenceDecision(
                allowed=False,
                reason="not_enough_results",
            )

        highest_vector_score = max(
            result.score
            for result in results
        )

        if highest_vector_score < self._minimum_score:
            return EvidenceDecision(
                allowed=False,
                reason="top_score_below_minimum",
            )
        
        return EvidenceDecision(
            allowed=True,
            reason="score_sufficient",
        )












