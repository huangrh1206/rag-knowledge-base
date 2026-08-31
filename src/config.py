from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

def parse_bool(
    value: str,
    name: str,
) -> bool:
    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be a boolean value"
    )


@dataclass(frozen=True)
class Settings:
    api_key:str
    base_url:str | None
    chat_model:str
    embedding_model:str
    embedding_batch_size: int = 20
    request_timeout: float = 30.0
    max_retries: int = 2
    chunk_size:int = 700
    chunk_overlap: int = 100
    top_k:int = 5
    retrieval_candidate_k: int = 20
    rerank_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    hybrid_enabled: bool = False
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    rrf_rank_constant: int = 60
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
            request_timeout=float(
                os.getenv("RAG_REQUEST_TIMEOUT", 30.0)
            ),
            max_retries=int(
                os.getenv("RAG_MAX_RETRIES", 2)
            ),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", 700)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", 100)),
            top_k=int(os.getenv("RAG_TOP_K", 5)),
            retrieval_candidate_k=int(
                os.getenv("RAG_RETRIEVAL_CANDIDATE_K", 20)
            ),
            rerank_enabled=parse_bool(
                os.getenv("RAG_RERANK_ENABLED", "false"),
                "RAG_RERANK_ENABLED",
            ),
            reranker_model=os.getenv(
                "RAG_RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            ).strip(),
            hybrid_enabled=parse_bool(
                os.getenv("RAG_HYBRID_ENABLED", "false"),
                "RAG_HYBRID_ENABLED",
            ),
            dense_weight=float(
                os.getenv("RAG_DENSE_WEIGHT", 1.0)
            ),
            bm25_weight=float(
                os.getenv("RAG_BM25_WEIGHT", 1.0)
            ),
            rrf_rank_constant=int(
                os.getenv("RAG_RRF_K", 60)
            ),            
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

        if settings.retrieval_candidate_k <= 0:
            raise ValueError(
                "retrieval candidate k must be positive."
            )

        if settings.retrieval_candidate_k < settings.top_k:
            raise ValueError(
                "retrieval candidate k must be greater than or equal to top k."
            )

        if not settings.reranker_model:
            raise ValueError(
                "reranker model cannot be empty."
            )

        if settings.dense_weight < 0:
            raise ValueError("dense weight must be non-negative")

        if settings.bm25_weight < 0:
            raise ValueError("bm25 weight must be non-negative")

        if settings.dense_weight == 0 and settings.bm25_weight == 0:
            raise ValueError(
                "at least one fusion weight must be positive"
            )

        if settings.rrf_rank_constant <= 0:
            raise ValueError(
                "RRF rank constant must be positive"
            )
        
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

        if settings.request_timeout <= 0:
            raise ValueError(
                "request timeout must be positive."
            )

        if settings.max_retries < 0:
            raise ValueError(
                "max retries must be non-negative."
            )
        
        return settings
    
