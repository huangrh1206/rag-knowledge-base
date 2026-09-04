from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
TEST_ROOT = PROJECT_ROOT / "tests"


def test_source_root_contains_only_composition_modules() -> None:
    root_modules = {
        path.name
        for path in SOURCE_ROOT.glob("*.py")
    }

    assert root_modules == {
        "__init__.py",
        "cli.py",
        "config.py",
    }


def test_source_is_grouped_into_responsibility_packages() -> None:
    expected_packages = {
        "agent",
        "infrastructure",
        "mcp",
        "persistence",
        "rag",
        "retrieval",
    }
    actual_packages = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }

    assert expected_packages <= actual_packages


def test_tests_are_grouped_by_source_responsibility() -> None:
    root_test_modules = set(TEST_ROOT.glob("test_*.py"))
    test_groups = {
        path.name
        for path in TEST_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert root_test_modules == set()
    assert test_groups == {
        "agent",
        "application",
        "evaluation",
        "infrastructure",
        "mcp",
        "persistence",
        "rag",
        "retrieval",
    }
