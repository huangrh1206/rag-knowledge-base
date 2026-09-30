"""Uvicorn entry points for the production API."""

from src.api.factory import create_production_app


def create_application():
    """Build the application lazily for Uvicorn's factory mode."""

    return create_production_app()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError(
            "API startup requires the optional 'api' dependencies"
        ) from exc
    uvicorn.run(
        "src.api.main:create_application",
        factory=True,
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    main()
