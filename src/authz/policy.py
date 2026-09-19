"""Authorization, credential lookup, and safe audit payload helpers."""

from dataclasses import dataclass
import os
import re
from typing import Any, Protocol


_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SENSITIVE_KEYS = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
)


@dataclass(frozen=True)
class AccessContext:
    tenant_id: str
    subject_id: str
    permissions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _validate_reference(self.tenant_id, "tenant id")
        _validate_reference(self.subject_id, "subject id")


class CredentialProvider(Protocol):
    def get(self, tenant_id: str, credential_ref: str) -> str:
        ...


class EnvironmentCredentialProvider:
    """Resolve opaque credential references from process environment."""

    def __init__(self, prefix: str = "RAG_CREDENTIAL") -> None:
        self._prefix = prefix.strip().upper()
        if not self._prefix:
            raise ValueError("credential environment prefix cannot be empty")

    def get(self, tenant_id: str, credential_ref: str) -> str:
        _validate_reference(tenant_id, "tenant id")
        _validate_reference(credential_ref, "credential reference")
        key = "_".join(
            (
                self._prefix,
                tenant_id.replace("-", "_").upper(),
                credential_ref.replace("-", "_").upper(),
            )
        )
        value = os.getenv(key, "")
        if not value:
            raise KeyError(f"credential is not configured: {credential_ref}")
        return value


class ToolAccessPolicy:
    def authorize(
        self,
        context: AccessContext,
        required_permissions: frozenset[str],
    ) -> None:
        missing = required_permissions - context.permissions
        if missing:
            names = ", ".join(sorted(missing))
            raise PermissionError(f"missing tool permissions: {names}")


def redact_sensitive(value: Any) -> Any:
    """Return a recursively redacted copy suitable for audit events."""

    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if str(key).lower() in _SENSITIVE_KEYS
                else redact_sensitive(nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    return value


def _validate_reference(value: str, label: str) -> None:
    if not isinstance(value, str) or not _REFERENCE.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, numbers, _ or -")
