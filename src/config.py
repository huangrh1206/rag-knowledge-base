from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

@dataclass(frozen=True)
class Settings:
    api_key:str
    base_url:str | None
    chat_model:str
    embedding_model:str
    embedding_batch_size: int = 20
    chunk_size:int = 700
    chunk_overlap: int = 100
    top_k:int = 5
    similarity_threshold:float = 0.30
    evidence_minimum_score: float = 0.50
    evidence_minimum_results: int = 1
    index_dir:Path = Path("storage/index")
    vector_store_backend: str = "numpy"
    qdrant_path: Path = Path("storage/qdrant")
    qdrant_collection: str = "rag_chunks"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        api_key = os.getenv("RAG_API_KEY", "").strip()
        if not api_key:
            raise ValueError("RAG_API_KEY is required.")
        
        settings = cls(
            api_key=api_key,
            base_url=os.getenv("RAG_BASE_URL") or None,
            chat_model=os.getenv(
                "RAG_CHAT_MODEL", 
                "gpt-4.1-mini"
            ),
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", 
                "text-embedding-3-small"
            ),
            embedding_batch_size=int(
                os.getenv("RAG_EMBEDDING_BATCH_SIZE", 20)
            ),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", 700)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", 100)),
            top_k=int(os.getenv("RAG_TOP_K", 5)),
            similarity_threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", 0.30)),
            evidence_minimum_score=float(os.getenv("RAG_EVIDENCE_MIN_SCORE", 0.50)),
            evidence_minimum_results=int(os.getenv("RAG_EVIDENCE_MIN_RESULTS", 1)),
            index_dir=Path(os.getenv("RAG_INDEX_DIR", "storage/index")),
            vector_store_backend=os.getenv("RAG_VECTOR_STORE_BACKEND", "numpy").strip().lower(),
            qdrant_path=Path(os.getenv("RAG_QDRANT_PATH", "storage/qdrant")),
            qdrant_collection=os.getenv("RAG_QDRANT_COLLECTION", "rag_chunks").strip(),
        )

        if settings.chunk_size <= 0:
            raise ValueError("chunk size must be positive.")

        if settings.embedding_batch_size <= 0:
            raise ValueError("embedding batch size must be positive.")
        
        if not 0 <= settings.chunk_overlap < settings.chunk_size:
            raise ValueError("overlap must be non-negative and less than chunk size.")
        
        if settings.top_k <= 0:
            raise ValueError("top k must be positive.")
        
        if not -1.0 <= settings.similarity_threshold <= 1.0:
            raise ValueError("RAG_SIMILARITY_THRESHOLD must be between -1.0 and 1.0.")

        if not 0.0 <= settings.evidence_minimum_score <= 1.0:
            raise ValueError(
                "evidence minimum score must be between 0 and 1."
            )

        if settings.evidence_minimum_results <= 0:
            raise ValueError(
                "evidence minimum results must be positive."
            )

        if settings.vector_store_backend not in {"numpy", "qdrant"}:
            raise ValueError(
                "vector store backend must be numpy or qdrant."
            )

        if not settings.qdrant_collection:
            raise ValueError(
                "qdrant collection cannot be empty"
            )
        
        return settings
    
