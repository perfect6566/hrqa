"""MCP client for FastMCP server.

Uses simple HTTP requests to call FastMCP's custom HTTP endpoints.
FastMCP's main /mcp endpoint requires full MCP protocol (session-based).
For simple JSON-RPC over HTTP, we use the /mcp-api endpoint.
"""

import asyncio
import json
import httpx
from typing import Any, Dict, List, Optional


class MCPClient:
    """Async MCP client for calling FastMCP tools."""

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        timeout: float = 30.0
    ):
        """Initialize MCP client.

        Args:
            server_url: URL of the MCP server (e.g., http://localhost:8001)
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._tools: Optional[List[Dict]] = None

    async def _request(self, method: str, params: Dict = None) -> Dict:
        """Make MCP request via /mcp-api endpoint."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": 1
        }
        if params:
            payload["params"] = params

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.server_url}/mcp-api",
                json=payload
            )
            response.raise_for_status()
            return response.json()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools from the MCP server."""
        result = await self._request("tools/list")
        tools = result.get("result", {}).get("tools", [])
        self._tools = tools
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call an MCP tool with arguments."""
        result = await self._request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        return result.get("result", {})

    async def health_check(self) -> Dict[str, Any]:
        """Check if MCP server is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/health")
                return {
                    "connected": True,
                    "status": response.status_code,
                    "data": response.json()
                }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def get_tools_schema(self) -> Dict[str, Any]:
        """Get the schema of all tools for LLM function calling."""
        if not self._tools:
            return {}

        schema = {}
        for tool in self._tools:
            schema[tool.get("name", "")] = {
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {})
            }

        return schema


class MCPClientSync:
    """Synchronous MCP client."""

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        timeout: float = 30.0
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._tools: Optional[List[Dict]] = None

    def _request(self, method: str, params: Dict = None) -> Dict:
        """Make MCP request via /mcp-api endpoint."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": 1
        }
        if params:
            payload["params"] = params

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.server_url}/mcp-api",
                json=payload
            )
            response.raise_for_status()
            return response.json()

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        result = self._request("tools/list")
        tools = result.get("result", {}).get("tools", [])
        self._tools = tools
        return tools

    def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call an MCP tool."""
        result = self._request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        return result.get("result", {})

    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.server_url}/health")
                return {
                    "connected": True,
                    "status": response.status_code,
                    "data": response.json()
                }
        except Exception as e:
            return {"connected": False, "error": str(e)}