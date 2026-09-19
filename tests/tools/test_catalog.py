import pytest

from src.authz import AccessContext, ToolAccessPolicy
from src.tools import ToolCatalog, ToolDescriptor, ToolSource, TrustLevel


class EchoTool:
    name = "echo"
    definition = {"type": "function", "function": {"name": "echo"}}

    def invoke(self, arguments: str) -> str:
        return arguments


def descriptor(version: str) -> ToolDescriptor:
    return ToolDescriptor(
        tool_id="utility.echo",
        version=version,
        source=ToolSource.LOCAL,
        trust=TrustLevel.TRUSTED,
        definition=EchoTool.definition,
        required_permissions=frozenset({"tools.invoke"}),
    )


def test_catalog_selects_latest_version_and_honors_tenant_disable() -> None:
    catalog = ToolCatalog()
    catalog.register(descriptor("1.0.0"), EchoTool())
    catalog.register(descriptor("1.2.0"), EchoTool())
    context = AccessContext(
        tenant_id="tenant-a",
        subject_id="user-1",
        permissions=frozenset({"tools.invoke"}),
    )

    selected = catalog.resolve(
        "utility.echo",
        context,
        ToolAccessPolicy(),
    )
    catalog.set_enabled("tenant-a", "utility.echo", False)

    assert selected.descriptor.version == "1.2.0"
    with pytest.raises(PermissionError, match="disabled"):
        catalog.resolve("utility.echo", context, ToolAccessPolicy())


def test_catalog_keeps_tenant_settings_isolated() -> None:
    catalog = ToolCatalog()
    catalog.register(descriptor("1.0.0"), EchoTool())
    policy = ToolAccessPolicy()
    tenant_a = AccessContext(
        "tenant-a", "user-a", frozenset({"tools.invoke"})
    )
    tenant_b = AccessContext(
        "tenant-b", "user-b", frozenset({"tools.invoke"})
    )
    catalog.set_enabled("tenant-a", "utility.echo", False)

    with pytest.raises(PermissionError):
        catalog.resolve("utility.echo", tenant_a, policy)
    assert catalog.resolve("utility.echo", tenant_b, policy).tool.name == "echo"


def test_catalog_requires_exact_registered_version_when_requested() -> None:
    catalog = ToolCatalog()
    catalog.register(descriptor("1.0.0"), EchoTool())
    context = AccessContext(
        "tenant", "user", frozenset({"tools.invoke"})
    )

    with pytest.raises(KeyError, match="2.0.0"):
        catalog.resolve(
            "utility.echo",
            context,
            ToolAccessPolicy(),
            version="2.0.0",
        )
