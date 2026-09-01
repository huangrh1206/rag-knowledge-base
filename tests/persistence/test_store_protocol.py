import numpy as np

from src.rag.models import Chunk, SearchResult
from src.persistence.protocol import SearchStore
from src.persistence.numpy_store import VectorStore

def test_numpy_vector_store_matches_search_store_protocol() -> None:
    store = VectorStore(
        [
            Chunk(
                id="guide-0000",
                text="content",
                source="guide.docx",
                paragraph_start=1,
                paragraph_end=1,
            )
        ],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    search_store: SearchStore = store

    results = search_store.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert isinstance(results[0], SearchResult)
    assert results[0].chunk.source == "guide.docx"
