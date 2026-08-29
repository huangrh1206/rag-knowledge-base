from openai import OpenAI

from src.config import Settings

def create_openai_client(
    settings: Settings,      
) -> OpenAI:
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )

    