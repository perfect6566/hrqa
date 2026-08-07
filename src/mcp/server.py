"""MCP server implementation using FastMCP."""

import json
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
from functools import wraps


class Tool:
    """Represents an MCP tool."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def call(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments."""
        try:
            result = self.handler(**arguments)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class ToolRegistry:
    """Registry for MCP tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema
            }
            for t in self._tools.values()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return {"success": False, "error": f"Tool '{name}' not found"}
        return tool.call(arguments)


def tool(
    name: str,
    description: str,
    input_schema: Dict[str, Any]
):
    """Decorator to register a function as an MCP tool."""
    def decorator(func: Callable) -> Callable:
        # Attach metadata to the function
        func._mcp_tool = {
            "name": name,
            "description": description,
            "input_schema": input_schema
        }
        return func
    return decorator


class MCPServer:
    """
    MCP server that exposes HR tools.
    
    Supports multiple transport modes:
    - stdio: For local subprocess communication
    - http: For HTTP-based communication
    """

    def __init__(self, name: str = "hr-mcp-server"):
        self.name = name
        self.registry = ToolRegistry()
        self._tools: Dict[str, Callable] = {}

    def tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any]
    ):
        """Decorator to register a tool handler."""
        def decorator(func: Callable) -> Callable:
            tool_obj = Tool(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=func
            )
            self.registry.register(tool_obj)
            self._tools[name] = func

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper._mcp_tool = {
                "name": name,
                "description": description,
                "input_schema": input_schema
            }
            return wrapper
        return decorator

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return self.registry.list_tools()

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with arguments."""
        return self.registry.call_tool(name, arguments)

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        method = request.get("method")
        params = request.get("params", {})

        if method == "tools/list":
            return {
                "result": {
                    "tools": self.list_tools()
                }
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = self.call_tool(tool_name, arguments)
            return {"result": result}
        else:
            return {
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}"
                }
            }

    def run_stdio(self):
        """Run server using stdio transport."""
        import sys

        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

    def get_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI spec for HTTP transport."""
        paths = {}
        for tool in self.list_tools():
            tool_name = tool["name"]
            paths[f"/tools/{tool_name}"] = {
                "post": {
                    "summary": f"Call {tool_name} tool",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": tool["inputSchema"]
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Tool result",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    }
                }
            }

        return {
            "openapi": "3.0.0",
            "info": {"title": self.name, "version": "1.0.0"},
            "paths": paths
        }
