from src.reranker import CrossEncoderReranker, DisabledReranker
from src.config import Settings
import logging

def reranker(settings: Settings):
    if not settings.rerank_enabled:
        return DisabledReranker()

    try:
        return CrossEncoderReranker(
            model_name=settings.reranker_model,
        )
    except (ImportError, RuntimeError, OSError) as exc:
        logging.getLogger(__name__).warning(
            "reranker unavailable; "
            "fallback to disabled reranker: %s",
            exc,
        )
        return DisabledReranker()