from __future__ import annotations
from typing import Any

class ApprovalRequired(Exception):
    def __init__(self, step_id: str, tool: str | None, arguments: dict[str, Any]):
        super().__init__(f"approval required for {step_id}"); self.step_id, self.tool, self.arguments = step_id, tool, arguments

def approval_payload(step_id: str, tool: str | None, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"step_id": step_id, "tool": tool, "arguments": arguments}

def resolve_approval(value: Any, arguments: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if value is True: return True, arguments
    if value is False or value is None: return False, arguments
    if isinstance(value, dict) and value.get("action") == "edit": return True, dict(value.get("arguments", arguments))
    return bool(value), arguments

