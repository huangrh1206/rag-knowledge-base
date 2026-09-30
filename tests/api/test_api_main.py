from unittest.mock import patch

from src.api.main import create_application


def test_create_application_defers_to_production_factory() -> None:
    application = object()

    with patch(
        "src.api.main.create_production_app",
        return_value=application,
    ) as factory:
        assert create_application() is application

    factory.assert_called_once_with()
