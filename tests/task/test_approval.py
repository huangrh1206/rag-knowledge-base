from src.task.approval import resolve_approval

def test_approval_edit():
    ok, args = resolve_approval({"action": "edit", "arguments": {"x": 2}}, {"x": 1})
    assert ok and args == {"x": 2}

