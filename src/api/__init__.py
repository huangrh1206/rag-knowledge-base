"""HTTP API composition layer."""

from src.api.app import AskRequest, AskResponse, create_app

def __getattr__(name: str):
    if name in {"create_agent", "create_production_app"}:
        from src.api.factory import create_agent, create_production_app
        return {"create_agent": create_agent, "create_production_app": create_production_app}[name]
    raise AttributeError(name)

__all__ = [
    "AskRequest",
    "AskResponse",
    "create_agent",
    "create_app",
    "create_production_app",
]
