import pytest

from src.workflow import NodeKind, WorkflowDefinition, migrate_workflow


def test_workflow_schema_parses_platform_nodes() -> None:
    workflow = WorkflowDefinition.from_dict(
        {
            "schema_version": 1,
            "workflow_id": "support-agent",
            "nodes": [
                {"id": "input", "kind": "input", "config": {}},
                {
                    "id": "search",
                    "kind": "tool",
                    "config": {
                        "tool_id": "knowledge.search",
                        "tool_version": "1.0.0",
                        "input_mapping": {"query": "$.question"},
                    },
                },
                {"id": "answer", "kind": "llm", "config": {}},
            ],
            "edges": [
                {"source": "input", "target": "search"},
                {"source": "search", "target": "answer"},
            ],
        }
    )

    assert workflow.nodes[1].kind is NodeKind.TOOL
    assert workflow.nodes[1].config["tool_id"] == "knowledge.search"


def test_workflow_schema_rejects_unsafe_or_invalid_config() -> None:
    with pytest.raises(ValueError, match="python_path"):
        WorkflowDefinition.from_dict(
            {
                "schema_version": 1,
                "workflow_id": "unsafe",
                "nodes": [
                    {
                        "id": "custom",
                        "kind": "tool",
                        "config": {
                            "tool_id": "custom.run",
                            "python_path": "package.module:function",
                        },
                    }
                ],
                "edges": [],
            }
        )

    with pytest.raises(ValueError, match="max_iterations"):
        WorkflowDefinition.from_dict(
            {
                "schema_version": 1,
                "workflow_id": "loop",
                "nodes": [
                    {
                        "id": "loop",
                        "kind": "loop",
                        "config": {"max_iterations": 0},
                    }
                ],
                "edges": [],
            }
        )


def test_version_zero_workflow_migrates_to_current_schema() -> None:
    migrated = migrate_workflow(
        {
            "version": 0,
            "id": "legacy",
            "nodes": [{"id": "start", "type": "input", "config": {}}],
            "edges": [],
        }
    )

    assert migrated["schema_version"] == 1
    assert migrated["workflow_id"] == "legacy"
    assert migrated["nodes"][0]["kind"] == "input"
