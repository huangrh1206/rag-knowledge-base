import json
from types import SimpleNamespace

import pytest

from src.agent.tools import RAGSearchTool, ToolRegistry
from src.rag.models import Chunk, SearchResult


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, question, top_k):
        self.calls.append((question, top_k))
        return [SearchResult(Chunk("c1", "evidence", "guide.docx", 1, 1), 0.9)][:top_k]


def test_rag_tool_validates_and_forwards_top_k():
    retriever = FakeRetriever()
    result = RAGSearchTool(retriever).invoke('{"query":"FastAPI", "top_k":3}')
    assert "evidence" in result
    assert retriever.calls == [("FastAPI", 3)]


@pytest.mark.parametrize("arguments", ["not-json", "[]", '{"query":""}', '{"query":"q", "top_k":0}', '{"query":"q", "top_k":11}', '{"query":"q", "unknown":true}'])
def test_rag_tool_returns_structured_validation_error(arguments):
    result = RAGSearchTool(FakeRetriever()).invoke(arguments)
    assert json.loads(result).keys() == {"error"}


def test_registry_rejects_duplicate_names():
    tool = RAGSearchTool(FakeRetriever())
    registry = ToolRegistry([tool])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_registry_discovers_and_invokes_tools():
    retriever = FakeRetriever()
    registry = ToolRegistry([RAGSearchTool(retriever)])
    assert registry.definitions()[0]["function"]["name"] == "search_knowledge_base"
    assert "evidence" in registry.invoke("search_knowledge_base", '{"query":"q"}')
    assert json.loads(registry.invoke("missing", "{}"))["error"] == "unknown tool"
