"""Minimal Streamable HTTP MCP client for remote task tools."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


class StreamableHTTPMCPClient:
    def __init__(self, url: str, *, api_key: str | None = None, timeout: float = 30.0):
        if not url.strip():
            raise ValueError("MCP URL cannot be empty")
        self.url = self._with_key(url, api_key)
        self.timeout = timeout
        self.session_id: str | None = None
        self._request_id = 0
        self._initialized = False

    @staticmethod
    def _with_key(url: str, api_key: str | None) -> str:
        if not api_key or "key=" in url or "api_key=" in url:
            return url
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["key"] = api_key
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._request_id += 1
        payload = {"jsonrpc": "2.0", "id": self._request_id, "method": method}
        if params is not None:
            payload["params"] = params
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            session_id = response.headers.get("Mcp-Session-Id")
            if session_id:
                self.session_id = session_id
            body = response.read().decode("utf-8")
        return self._decode(body)

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout):
            return

    @staticmethod
    def _decode(body: str) -> Any:
        body = body.strip()
        if body.startswith("data:"):
            body = "\n".join(line[5:].strip() for line in body.splitlines() if line.startswith("data:"))
        value = json.loads(body)
        if "error" in value:
            raise RuntimeError(value["error"].get("message", "MCP request failed"))
        return value.get("result", value)

    def initialize(self) -> None:
        if self._initialized:
            return
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "general-task-agent", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", None)
        self._initialized = True

    def list_tools(self) -> list[dict[str, Any]]:
        self.initialize()
        result = self._rpc("tools/list", {})
        return list(result.get("tools", []))

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.initialize()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(f"MCP tool failed: {name}")
        content = result.get("content", result)
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            text = "\n".join(texts).strip()
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return content
