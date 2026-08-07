"""MCP server for HR tools using FastMCP (official MCP framework)."""

from .fastmcp_server import create_mcp_server, get_mcp_server, set_rag_pipeline, get_rag_pipeline, run_server
from .tools import HRTools

__all__ = [
    "create_mcp_server",
    "get_mcp_server",
    "set_rag_pipeline",
    "get_rag_pipeline",
    "run_server",
    "HRTools",
]
