"""MCP client for calling tools via Model Context Protocol."""

import json
import httpx
from typing import Any, Dict, List, Optional


class MCPClient:
    """Client for calling MCP server tools."""

    def __init__(
        self,
        server_url: str = "http://localhost:8001",
        timeout: float = 30.0
    ):
        """
        Initialize MCP client.

        Args:
            server_url: URL of the MCP server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._tools: Optional[List[Dict]] = None

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools from the MCP server.

        Returns:
            List of tool definitions
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.server_url}/mcp",
                json={"method": "tools/list"}
            )
            response.raise_for_status()
            data = response.json()
            self._tools = data.get("result", {}).get("tools", [])
            return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call an MCP tool with arguments.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.server_url}/mcp",
                json={
                    "method": "tools/call",
                    "params": {
                        "name": name,
                        "arguments": arguments
                    }
                }
            )
            response.raise_for_status()
            return response.json().get("result", {})

    async def health_check(self) -> Dict[str, Any]:
        """Check if MCP server is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.server_url}/health")
                return {"connected": True, "status": response.status_code}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def get_tools_schema(self) -> Dict[str, Any]:
        """Get the schema of all tools for LLM function calling."""
        if not self._tools:
            return {}

        schema = {}
        for tool in self._tools:
            schema[tool["name"]] = {
                "description": tool["description"],
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

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.server_url}/mcp",
                json={"method": "tools/list"}
            )
            response.raise_for_status()
            data = response.json()
            self._tools = data.get("result", {}).get("tools", [])
            return self._tools

    def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call an MCP tool."""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.server_url}/mcp",
                json={
                    "method": "tools/call",
                    "params": {
                        "name": name,
                        "arguments": arguments
                    }
                }
            )
            response.raise_for_status()
            return response.json().get("result", {})

    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.server_url}/health")
                return {"connected": True, "status": response.status_code}
        except Exception as e:
            return {"connected": False, "error": str(e)}
