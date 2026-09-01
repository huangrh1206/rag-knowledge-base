from typing import Protocol

import numpy as np

from src.models import Chunk, SearchResult

class SearchStore(Protocol):
    @property
    def vector_dimension(self) -> int:
        ...

    def all_chunks(self) -> list[Chunk]:
        """Return all indexed chunks for lexical retrieval."""
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

    








