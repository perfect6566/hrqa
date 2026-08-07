"""FastAPI web application for HR Policy Assistant with MCP integration."""

import os
import asyncio
import multiprocessing
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import threading
import time

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env", override=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag import RAGPipeline
from src.agent import AgentOrchestrator
from src.mcp_server import fastmcp_client as mcp_client_module


# =============================================================================
# Global Instances
# =============================================================================

rag_pipeline: Optional[RAGPipeline] = None
orchestrator: Optional[AgentOrchestrator] = None
mcp_server_process: Optional[multiprocessing.Process] = None
mcp_client: Optional[mcp_client_module.MCPClient] = None


def get_orchestrator() -> Optional[AgentOrchestrator]:
    """Get the orchestrator instance."""
    return orchestrator


def get_mcp_client() -> Optional[mcp_client_module.MCPClient]:
    """Get the MCP client instance."""
    return mcp_client


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(default="user")
    content: str


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User's message")
    employee_id: Optional[str] = Field(None, description="Employee ID if known")
    history: Optional[List[ChatMessage]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    app_status: str
    mcp_connected: bool
    index_status: str
    tools_count: Optional[int] = None
    mcp_protocol_used: bool = True


# =============================================================================
# MCP Server Management
# =============================================================================

def run_mcp_server(port: int):
    """Run MCP server in a separate process using FastMCP."""
    from src.mcp_server.app import create_app

    mcp = create_app()
    mcp.run(transport="streamable-http", port=port, host="127.0.0.1")


def start_mcp_server(port: int = 8001) -> bool:
    """Start MCP server as a subprocess."""
    global mcp_server_process
    
    try:
        # Check if MCP server is already running
        import httpx
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"MCP server already running on port {port}")
                return True
        except:
            pass
        
        # Start MCP server in background process
        mcp_server_process = multiprocessing.Process(
            target=run_mcp_server,
            args=(port,),
            daemon=True
        )
        mcp_server_process.start()
        
        # Wait for server to start
        for _ in range(30):  # 30 attempts, 0.5s each = 15s max wait
            time.sleep(0.5)
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if response.status_code == 200:
                    print(f"MCP server started successfully on port {port}")
                    return True
            except:
                continue
        
        print("Warning: MCP server may not have started properly")
        return True  # Continue anyway, will attempt direct fallback
        
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        return True  # Continue anyway


def stop_mcp_server():
    """Stop MCP server."""
    global mcp_server_process
    
    if mcp_server_process and mcp_server_process.is_alive():
        mcp_server_process.terminate()
        mcp_server_process.join(timeout=5)
        print("MCP server stopped")


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global rag_pipeline, orchestrator, mcp_client
    
    # Startup
    print("=" * 60)
    print("Starting HR Policy Assistant with MCP Integration...")
    print("=" * 60)
    
    print(f"OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'NOT SET')}")
    
    # Get MCP server URL
    mcp_port = int(os.getenv("MCP_PORT", "8001"))
    mcp_server_url = f"http://127.0.0.1:{mcp_port}"
    
    # Start MCP server as subprocess
    print(f"Starting MCP server on port {mcp_port}...")
    mcp_started = start_mcp_server(mcp_port)
    print(f"MCP server status: {'Running' if mcp_started else 'Failed to start'}")
    
    # Initialize MCP client for health checks
    mcp_client = mcp_client_module.MCPClient(server_url=mcp_server_url)
    
    # Initialize RAG pipeline
    policies_dir = os.getenv("POLICIES_DIR", "policies")
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")

    try:
        print("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(
            policies_dir=policies_dir,
            vector_store_path=vector_store_path,
            embedder_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            generator_model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            chunk_size=512,
            chunk_overlap=50,
            top_k=5,
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        # Index documents
        print("Indexing policy documents...")
        index_result = rag_pipeline.index_documents()
        print(f"Indexing complete: {index_result}")
        
        # Connect RAG pipeline to MCP server BEFORE starting MCP
        print("Connecting RAG pipeline to MCP server...")
        from src.mcp_server import fastmcp_server
        fastmcp_server.set_rag_pipeline(rag_pipeline)
        print("RAG pipeline connected to MCP server")

    except Exception as e:
        print(f"Error initializing RAG: {e}")
        rag_pipeline = None

    # Initialize agent orchestrator with MCP protocol
    try:
        print("Initializing agent orchestrator with MCP protocol...")
        orchestrator = AgentOrchestrator(
            rag_pipeline=rag_pipeline,
            mcp_server_url=mcp_server_url,  # Use actual MCP server URL
            model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
            use_mcp_protocol=True  # Ensure MCP protocol is used
        )
        print("Agent orchestrator initialized with MCP protocol")
        
        # Verify MCP connection
        try:
            health = await mcp_client.health_check()
            if health.get("connected"):
                tools = await mcp_client.list_tools()
                print(f"MCP connected: {len(tools)} tools available")
        except Exception as e:
            print(f"MCP connection verification failed: {e}")
            
    except Exception as e:
        print(f"Error initializing orchestrator: {e}")
        orchestrator = None

    print("=" * 60)
    print("HR Policy Assistant started successfully!")
    print("=" * 60)

    yield

    # Shutdown
    print("Shutting down HR Policy Assistant...")
    stop_mcp_server()
    print("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="HR Policy Assistant",
    description="Agentic AI system for HR policy and operations tasks with MCP integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    mcp_connected = False
    tools_count = None

    if mcp_client:
        try:
            health = await mcp_client.health_check()
            mcp_connected = health.get("connected", False)
            if mcp_connected:
                tools = await mcp_client.list_tools()
                tools_count = len(tools)
        except Exception as e:
            print(f"Health check error: {e}")

    index_status = "not_ready"
    if rag_pipeline:
        stats = rag_pipeline.get_index_stats()
        if stats.get("status") == "indexed":
            index_status = f"ready ({stats.get('chunks', 0)} chunks)"

    return HealthResponse(
        status="healthy",
        app_status="running",
        mcp_connected=mcp_connected,
        index_status=index_status,
        tools_count=tools_count,
        mcp_protocol_used=True
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for HR policy questions.

    Processes user messages and returns agentic responses with citations and traces.
    Tools are called via MCP protocol.
    """
    if not orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Please try again."
        )

    try:
        # Convert history to format expected by orchestrator
        history = None
        if request.history:
            history = [{"role": m.role, "content": m.content} for m in request.history]

        # Process request (uses MCP protocol internally)
        result = await orchestrator.process_request(
            query=request.message,
            employee_id=request.employee_id,
            conversation_history=history
        )

        return ChatResponse(
            answer=result["answer"],
            citations=result.get("citations", []),
            tool_calls=result.get("tool_calls", []),
            trace=result.get("trace", []),
            metadata=result.get("metadata", {})
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.get("/chat/history")
async def get_history():
    """Get empty history (placeholder for session management)."""
    return {"history": []}


@app.get("/capabilities")
async def get_capabilities():
    """Get agent capabilities."""
    if not orchestrator:
        return {"error": "Agent not initialized"}

    return orchestrator.get_capabilities()


@app.get("/mcp/status")
async def mcp_status():
    """Get MCP server status and tool list."""
    if not mcp_client:
        return {"connected": False, "error": "MCP client not initialized"}

    try:
        health = await mcp_client.health_check()
        tools = await mcp_client.list_tools()
        return {
            "connected": health.get("connected", False),
            "tools_count": len(tools),
            "tools": tools,
            "server_url": mcp_client.server_url
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.get("/employees")
async def list_employees():
    """List active employees for the frontend picker."""
    import json
    from pathlib import Path
    data_path = Path(__file__).parent.parent.parent / "mock_data" / "employees.json"
    try:
        payload = json.loads(data_path.read_text(encoding="utf-8"))
        employees = payload.get("employees", payload) if isinstance(payload, dict) else payload
        return {
            "employees": [
                {
                    "employee_id": e["employee_id"],
                    "name": e["name"],
                    "department": e.get("department"),
                    "office_location": e.get("office_location"),
                }
                for e in employees
                if e.get("employment_status") == "active"
            ]
        }
    except FileNotFoundError:
        return {"employees": []}
    except Exception as e:
        return {"employees": [], "error": str(e)}


# =============================================================================
# Demo Tasks Endpoints
# =============================================================================

@app.get("/demo/pto-request")
async def demo_pto_request():
    """
    Demo: PTO Request Guidance workflow.

    This demonstrates the agent checking an employee's PTO balance,
    retrieving PTO policy, and providing guidance via MCP.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    demo_query = "Can I take 3 days of PTO next week?"

    result = await orchestrator.process_request(
        query=demo_query,
        employee_id="EMP001"  # Alice Johnson
    )

    return {
        "task": "PTO Request Guidance",
        "query": demo_query,
        "response": result
    }


@app.get("/demo/remote-work")
async def demo_remote_work():
    """
    Demo: Remote Work Eligibility workflow.

    This demonstrates the agent checking employee profile,
    retrieving remote work policies, and providing compliance info via MCP.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    demo_query = "Can I work remotely from another state for 6 weeks?"

    result = await orchestrator.process_request(
        query=demo_query,
        employee_id="EMP002"  # Bob Smith
    )

    return {
        "task": "Remote Work Eligibility",
        "query": demo_query,
        "response": result
    }


# =============================================================================
# Web UI
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    from pathlib import Path
    html_path = Path(__file__).parent / "static" / "index.html"
    response = HTMLResponse(html_path.read_text(encoding="utf-8"))
    # Disable caching so users always see the latest UI updates
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# =============================================================================
# Policy Document Viewer
# =============================================================================

# IMPORTANT: declare /policy/section BEFORE the /policy/{filename} wildcard
# so FastAPI's routing does not capture "section" as a filename.
@app.get("/policy/section")
async def get_policy_section(document_id: str, section: str):
    """
    Return the chunks for a specific policy document and section heading.
    Used by the frontend to open a focused view of one referenced source.
    """
    import json
    from pathlib import Path

    if not document_id or not section:
        raise HTTPException(status_code=400, detail="document_id and section are required")

    # Path-traversal guard
    safe_doc = Path(document_id).name
    vector_store_path = Path(__file__).parent.parent.parent / "data" / "vector_store" / "chunks.json"
    if not vector_store_path.exists():
        raise HTTPException(status_code=503, detail="Index not ready")

    chunks = json.loads(vector_store_path.read_text(encoding="utf-8"))

    # Find chunks matching the document + section (heading starts with the section title)
    section_lower = section.lower().strip()
    matches = []
    seen = set()
    for c in chunks:
        if c.get("document_id") != safe_doc:
            continue
        heading = (c.get("heading") or "").split(" / ")[0].strip().lower()
        if heading == section_lower:
            key = c.get("id")
            if key in seen:
                continue
            seen.add(key)
            matches.append({
                "id": c.get("id"),
                "heading": c.get("heading"),
                "content": c.get("content"),
                "filename": (c.get("metadata") or {}).get("filename", safe_doc),
            })

    if not matches:
        # Fallback: match by partial heading (in case the section is a fragment)
        for c in chunks:
            if c.get("document_id") != safe_doc:
                continue
            heading = (c.get("heading") or "").split(" / ")[0].strip().lower()
            if section_lower in heading or heading in section_lower:
                key = c.get("id")
                if key in seen:
                    continue
                seen.add(key)
                matches.append({
                    "id": c.get("id"),
                    "heading": c.get("heading"),
                    "content": c.get("content"),
                    "filename": (c.get("metadata") or {}).get("filename", safe_doc),
                })

    # Load the full document to render the modal
    policies_dir = Path(__file__).parent.parent.parent / "policies"
    candidates = [
        policies_dir / f"{safe_doc}.md",
        policies_dir / f"{safe_doc}.html",
        policies_dir / f"{safe_doc}.pdf",
        policies_dir / f"{safe_doc}.txt",
    ]
    doc_path = next((p for p in candidates if p.exists()), None)
    if not doc_path:
        raise HTTPException(status_code=404, detail=f"Policy '{safe_doc}' not found")

    if doc_path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(doc_path)
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages_text.append(f"[Page {i + 1}]\n{text}")
        full_content = "\n\n".join(pages_text)
        doc_title = (reader.metadata.get("/Title") if reader.metadata else None) or doc_path.stem.replace("-", " ").title()
        full_html = f"<pre style='white-space:pre-wrap;word-break:break-word;'>{full_content}</pre>"
    else:
        full_content = doc_path.read_text(encoding="utf-8")
        doc_title = full_content.split("\n", 1)[0].lstrip("#").strip() or doc_path.stem.replace("-", " ").title()
        full_html = markdown_to_html(full_content)

    # If chunks did not contain the exact section, extract a focused snippet
    # from the raw document by searching for the section heading pattern.
    if not matches:
        section_content = extract_section_from_text(full_content, section, doc_path.suffix)
    else:
        section_content = "\n\n---\n\n".join(m["content"] for m in matches[:2])

    if not section_content:
        # Final fallback: show the first chunk of the document so the user
        # still gets something useful when the section name is unrecognised.
        section_content = full_content[:1500]

    return {
        "document_id": safe_doc,
        "section": section,
        "filename": doc_path.name,
        "title": doc_title,
        "section_content": section_content,
        "full_content": full_content,
        "full_html": full_html,
        "matched_chunks": matches,
    }


@app.get("/policy/{filename}")
async def get_policy(filename: str):
    """
    Serve a policy document for the frontend viewer modal.
    filename: the basename of the policy file (e.g. 'pto-policy' or 'pto-policy.md')
    """
    from pathlib import Path

    policies_dir = Path(__file__).parent.parent.parent / "policies"
    # Strip any path separators to prevent directory traversal
    safe_name = Path(filename.replace("/", "").replace("\\", "")).stem
    candidates = [
        policies_dir / f"{safe_name}.md",
        policies_dir / f"{safe_name}.txt",
        policies_dir / f"{safe_name}.html",
        policies_dir / f"{safe_name}.pdf",
    ]
    for path in candidates:
        if path.exists():
            # Handle PDF files
            if path.suffix.lower() == '.pdf':
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(path)
                    pages_text = []
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            pages_text.append(f"[Page {page_num + 1}]\n{text}")
                    content = "\n\n".join(pages_text)
                    # Get title from metadata or filename
                    title = None
                    if reader.metadata and reader.metadata.get("/Title"):
                        title = reader.metadata.get("/Title")
                    if not title:
                        title = path.stem.replace("-", " ").replace("_", " ").title()
                    # Convert PDF text to simple HTML
                    html = f"<pre style='white-space:pre-wrap;word-break:break-word;'>{content}</pre>"
                    return {
                        "filename": path.name,
                        "title": title,
                        "content": content,
                        "html": html,
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

            # Handle text-based files
            content = path.read_text(encoding="utf-8")
            # Extract the first markdown heading as the title
            doc_title = None
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("#"):
                    doc_title = stripped.lstrip("#").strip()
                    break
            if not doc_title:
                doc_title = path.stem.replace("-", " ").replace("_", " ").title()
            html = markdown_to_html(content)
            return {
                "filename": path.name,
                "title": doc_title,
                "content": content,
                "html": html,
            }
    raise HTTPException(status_code=404, detail=f"Policy '{safe_name}' not found")


def markdown_to_html(md: str) -> str:
    """Minimal markdown-to-HTML converter for policy display."""
    import re
    html = md
    # Headings
    html = re.sub(r'^###### (.+)$', r'<h6>\1</h6>', html, flags=re.MULTILINE)
    html = re.sub(r'^##### (.+)$', r'<h5>\1</h5>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    # Bold / italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    # Lists
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', html, flags=re.MULTILINE)
    # Paragraphs
    html = re.sub(r'\n\n+', r'\n\n', html)
    paras = []
    for para in html.split("\n\n"):
        p = para.strip()
        if not p:
            continue
        if p.startswith("<h") or p.startswith("<li"):
            paras.append(p)
        else:
            paras.append(f"<p>{p.replace(chr(10), '<br>')}</p>")
    return "\n".join(paras)


def extract_section_from_text(content: str, section: str, suffix: str) -> str:
    """
    Pull a focused snippet from the raw document by locating the section
    heading and grabbing the lines following it until the next heading of
    the same or higher level.
    """
    import re
    section_lower = section.lower().strip()
    if not section_lower:
        return ""

    lines = content.split("\n")
    is_html = suffix.lower() in (".html", ".htm")
    is_pdf = suffix.lower() == ".pdf"

    # First locate the heading line that matches the requested section.
    start_idx = -1
    matched_level = None
    if is_html:
        # Use level number from the h-tag so we can decide when to stop.
        heading_re = re.compile(r'<h([1-6])[^>]*>(.*?)</h\1>', re.IGNORECASE)
    elif is_pdf:
        heading_re = re.compile(r'\[Page\s+\d+\]')
    else:
        heading_re = re.compile(r'^(#{1,6})\s+(.+)$')

    for i, line in enumerate(lines):
        m = heading_re.search(line) if (is_html or is_pdf) else heading_re.match(line)
        if not m:
            continue
        if is_html:
            text = re.sub(r'<[^>]+>', '', m.group(2)).strip().lower()
        elif is_pdf:
            text = line.lower()
        else:
            text = m.group(2).strip().lower()

        if section_lower in text or text in section_lower:
            start_idx = i
            if is_html:
                matched_level = int(m.group(1))
            elif not is_pdf:
                matched_level = len(m.group(1))
            else:
                matched_level = 1
            break

    if start_idx < 0:
        return ""

    # Capture from start_idx until the next heading of the same or higher level.
    snippet_lines = []
    for j in range(start_idx, len(lines)):
        line = lines[j]
        if j > start_idx:
            m = heading_re.search(line) if (is_html or is_pdf) else heading_re.match(line)
            if m:
                if is_html:
                    level = int(m.group(1))
                elif is_pdf:
                    level = 1
                else:
                    level = len(m.group(1))
                # Stop when we hit a heading of the same or higher level
                # (i.e. smaller or equal level number). Sub-headings are kept.
                if level <= matched_level:
                    break
        snippet_lines.append(line)
        if len(snippet_lines) >= 80:
            snippet_lines.append("...")
            break

    return "\n".join(snippet_lines).strip()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
