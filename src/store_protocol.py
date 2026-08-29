from typing import Protocol

import numpy as np

from src.models import SearchResult

class SearchStore(Protocol):
    @property
    def vectory_dimension(self) -> int:
        ...
    # src\vector_store.py
    # src\qdrant_vector_store.py
    def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        """Return the highest-scoring results for one query."""
        ...

    








