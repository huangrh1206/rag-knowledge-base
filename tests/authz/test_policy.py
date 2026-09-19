import pytest

from src.authz import (
    AccessContext,
    EnvironmentCredentialProvider,
    ToolAccessPolicy,
    redact_sensitive,
)


def test_policy_rejects_missing_permission() -> None:
    context = AccessContext("tenant-a", "user-1", frozenset())

    with pytest.raises(PermissionError, match="tools.invoke"):
        ToolAccessPolicy().authorize(context, frozenset({"tools.invoke"}))


def test_environment_credentials_use_reference_not_secret(monkeypatch) -> None:
    monkeypatch.setenv("RAG_CREDENTIAL_TENANT_A_WEATHER", "secret-value")
    provider = EnvironmentCredentialProvider()

    assert provider.get("tenant-a", "weather") == "secret-value"
    with pytest.raises(ValueError, match="credential reference"):
        provider.get("tenant-a", "../weather")


def test_sensitive_values_are_redacted_recursively() -> None:
    value = {
        "api_key": "abc",
        "request": {"token": "xyz", "query": "weather"},
    }

    assert redact_sensitive(value) == {
        "api_key": "[REDACTED]",
        "request": {"token": "[REDACTED]", "query": "weather"},
    }
