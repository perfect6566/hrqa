"""MCP Server application with FastMCP.

This module uses FastMCP, the official recommended framework for MCP servers,
providing standards-compliant MCP protocol implementation.
"""

import os
import sys
from typing import Optional
from pathlib import Path

from fastmcp import FastMCP
import uvicorn

from .fastmcp_server import create_mcp_server, set_rag_pipeline, get_rag_pipeline


# Global FastMCP server instance
_mcp_server: Optional[FastMCP] = None


def get_mcp_server() -> Optional[FastMCP]:
    """Get the FastMCP server instance."""
    return _mcp_server


def _initialize_rag_pipeline():
    """Initialize the RAG pipeline inside the MCP 8001 process.

    The MCP server runs in its own process (uvicorn spawns it), so the
    ``set_rag_pipeline`` call from the API 8000 process would only mutate
    that process's globals. We need our own RAG pipeline here so that
    ``search_policy_documents`` can actually retrieve chunks.

    The env vars used here mirror the ones the API 8000 process passes to
    its own RAG pipeline, so both processes load the same vector store.
    """
    try:
        # Make sure we can import ``src.rag`` from the package root.
        repo_root = Path(__file__).resolve().parents[2]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from src.rag.rag_pipeline import RAGPipeline

        policies_dir = os.getenv("POLICIES_DIR", "policies")
        vector_store_path = os.getenv(
            "VECTOR_STORE_PATH", "./data/vector_store"
        )
        embedder_model = os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        generator_model = os.getenv("OPENAI_MODEL", "deepseek-chat")

        print(f"[MCP] Initializing RAG pipeline (policies={policies_dir}, "
              f"vector_store={vector_store_path})...")
        pipeline = RAGPipeline(
            policies_dir=policies_dir,
            vector_store_path=vector_store_path,
            embedder_model=embedder_model,
            generator_model=generator_model,
            chunk_size=512,
            chunk_overlap=50,
            top_k=5,
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        # ``VectorStore.__init__`` already auto-loads ``index.faiss`` /
        # ``chunks.json`` from ``vector_store_path`` if they exist. The API
        # 8000 process indexed documents on its first run, so this is fast.
        store_path = Path(vector_store_path)
        meta_path = store_path / "chunks.json"
        if meta_path.exists():
            print(f"[MCP] Vector store loaded with "
                  f"{len(pipeline.vector_store.chunks)} chunks")
        else:
            print("[MCP] No existing vector store; indexing now...")
            pipeline.index_documents()
        return pipeline
    except Exception as e:
        print(f"[MCP] Failed to initialize RAG pipeline: {e}")
        return None


def initialize_server():
    """Initialize the FastMCP server with all tools."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = create_mcp_server()
        rag = _initialize_rag_pipeline()
        if rag is not None:
            set_rag_pipeline(rag)
            print("[MCP] RAG pipeline attached to FastMCP server")
        else:
            print("[MCP] WARNING: RAG pipeline not attached; "
                  "search_policy_documents will fail")
    return _mcp_server


def create_app() -> FastMCP:
    """Create and return the FastMCP application.

    This creates the FastMCP instance with all tools registered.
    The FastMCP instance itself acts as a FastAPI app when using HTTP transport.
    """
    return initialize_server()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    mcp = create_app()

    print(f"Starting FastMCP server on port {port}...")
    print("Available transports: streamable-http (default), stdio")
    print("MCP endpoint: /mcp")
    print("Tools endpoint: /tools/list")

    mcp.run(transport="streamable-http", port=port, host="127.0.0.1")
