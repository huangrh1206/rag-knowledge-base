import json

from src.rag.models import Chunk, SearchResult
from src.config import Settings

def load_bm25_results(
    settings: Settings,
) -> list[SearchResult]:
    metadata = json.loads(
        (
            settings.index_dir / "chunks.json"
        ).read_text(encoding="utf-8")
    )

    return [
        SearchResult(
            chunk=Chunk.from_dict(item),
            score=0.0,
        )
        for item in metadata
    ]
