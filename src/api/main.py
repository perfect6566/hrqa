"""FastAPI web application for HR Policy Assistant with MCP integration.

The worker boots in two phases:

1. **Lifespan** (called by uvicorn once at startup) only does *cheap*
   work that has to happen before ``listen()`` returns:

   * reading environment variables,
   * constructing the MCP client object (no I/O),
   * spawning a background ``asyncio.Task`` that pre-warms the heavy
     components (RAG pipeline + MCP server subprocess + agent
     orchestrator).

   ``/health`` becomes available within a few hundred milliseconds, so
   Render's health probe never times out — even on the free tier where
   cold-start can be slow.

2. **First request** that actually needs the RAG / MCP / orchestrator
   awaits ``ensure_initialized()``, which guarantees a single
   initialization across concurrent first-callers. Subsequent requests
   hit the already-built globals.

If the background pre-warm fails or is still running when the first
request arrives, ``ensure_initialized`` picks up the work transparently
with a bounded wait + clear error message.
"""

import os
import asyncio
import multiprocessing
import traceback
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# Path setup
# ----------------------------------------------------------------------------
# Load .env file from project root.
project_root = Path(__file__).parent.parent.parent

# IMPORTANT: do NOT pass ``override=True`` here.
#
# In production Render sets OPENAI_API_KEY / OPENAI_BASE_URL via the
# dashboard. ``override=True`` would let a stale or partial .env file
# (e.g. one checked in by accident) silently overwrite those values.
# Letting existing process env take precedence means the dashboard wins
# by default, which is the contract Render expects.
load_dotenv(project_root / ".env", override=False)

import sys
sys.path.insert(0, str(project_root))

from src.rag import RAGPipeline  # noqa: E402
from src.agent import AgentOrchestrator  # noqa: E402
from src.mcp_server import fastmcp_client as mcp_client_module  # noqa: E402


# =============================================================================
# Global Instances
# =============================================================================
# These start as ``None`` and are populated by ``initialize_heavy_components``
# on the first request (or earlier by the background pre-warm task).

rag_pipeline: Optional[RAGPipeline] = None
orchestrator: Optional[AgentOrchestrator] = None
mcp_server_process: Optional[multiprocessing.Process] = None
mcp_client: Optional[mcp_client_module.MCPClient] = None

# Concurrency control for first-request initialization. ``_init_event`` is
# set when initialization finishes (success or failure); concurrent
# callers wait on it instead of all racing to build the same components.
_init_event: Optional[asyncio.Event] = None
_init_lock: Optional[asyncio.Lock] = None
_init_error: Optional[BaseException] = None


def get_orchestrator() -> Optional[AgentOrchestrator]:
    """Return the orchestrator instance (may be ``None`` before init)."""
    return orchestrator


def get_mcp_client() -> Optional[mcp_client_module.MCPClient]:
    """Return the MCP client instance (may be ``None`` before init)."""
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
    """Liveness health check (cheap, always returns 200 once the worker is listening)."""
    status: str
    app_status: str
    initialized: bool


class ReadinessResponse(BaseModel):
    """Readiness check — deep status of heavy components."""
    status: str
    initialized: bool
    mcp_connected: bool
    index_status: str
    tools_count: Optional[int] = None
    init_error: Optional[str] = None


# =============================================================================
# MCP Server Management
# =============================================================================

def run_mcp_server(port: int):
    """Run MCP server in a separate process using FastMCP."""
    from src.mcp_server.app import create_app

    mcp = create_app()
    mcp.run(transport="streamable-http", port=port, host="127.0.0.1")


def start_mcp_server(port: int = 8001) -> bool:
    """Start MCP server as a subprocess. Blocking — run via ``asyncio.to_thread``."""
    global mcp_server_process

    try:
        import httpx
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"MCP server already running on port {port}")
                return True
        except Exception:
            pass

        mcp_server_process = multiprocessing.Process(
            target=run_mcp_server,
            args=(port,),
            daemon=True,
        )
        mcp_server_process.start()

        for _ in range(30):
            import time
            time.sleep(0.5)
            try:
                response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if response.status_code == 200:
                    print(f"MCP server started successfully on port {port}")
                    return True
            except Exception:
                continue

        print("Warning: MCP server may not have started properly")
        return True
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        return True


def stop_mcp_server():
    """Stop the MCP server subprocess."""
    global mcp_server_process
    if mcp_server_process and mcp_server_process.is_alive():
        mcp_server_process.terminate()
        mcp_server_process.join(timeout=5)
        print("MCP server stopped")


# =============================================================================
# Heavy component initialization (lazy, concurrency-safe)
# =============================================================================

def _init_heavy_components_sync() -> None:
    """Synchronous body of heavy-component initialization.

    Runs in a worker thread (via ``asyncio.to_thread``) so it does not
    block the event loop while it spawns the MCP subprocess and loads
    the RAG index. Wrapped in try/except so a failure can never crash
    the lifespan — the event is set regardless and ``_init_error``
    captures the cause for ``/ready`` to surface.
    """
    global rag_pipeline, orchestrator, mcp_client, _init_error

    mcp_port = int(os.getenv("MCP_PORT", "8001"))
    mcp_server_url = f"http://127.0.0.1:{mcp_port}"

    try:
        # MCP client construction is cheap (no network I/O).
        try:
            mcp_client = mcp_client_module.MCPClient(server_url=mcp_server_url)
        except Exception as e:
            print(f"[lazy-init] WARN: MCP client construction failed: {e}")
            mcp_client = None

        # Spawn MCP server subprocess.
        try:
            print(f"[lazy-init] Starting MCP server on port {mcp_port}...")
            start_mcp_server(mcp_port)
            print("[lazy-init] MCP server start attempted")
        except Exception as e:
            print(f"[lazy-init] WARN: start_mcp_server raised: {e}")

        # RAG pipeline (embedder model itself stays lazy — see embedder.py).
        policies_dir = os.getenv("POLICIES_DIR", "policies")
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")

        try:
            print("[lazy-init] Building RAG pipeline...")
            rag_pipeline = RAGPipeline(
                policies_dir=policies_dir,
                vector_store_path=vector_store_path,
                embedder_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                generator_model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
                chunk_size=512,
                chunk_overlap=50,
                top_k=5,
                base_url=os.getenv("OPENAI_BASE_URL"),
            )

            print("[lazy-init] Indexing policy documents...")
            index_result = rag_pipeline.index_documents()
            print(f"[lazy-init] Indexing complete: {index_result}")

            # Wire the RAG pipeline into the MCP server (separate process)
            # so the FastMCP ``search_policy_documents`` tool can use it.
            try:
                from src.mcp_server import fastmcp_server
                fastmcp_server.set_rag_pipeline(rag_pipeline)
                print("[lazy-init] RAG pipeline connected to MCP server")
            except Exception as e:
                print(f"[lazy-init] WARN: failed to attach RAG to MCP: {e}")
        except Exception as e:
            print(f"[lazy-init] ERROR: RAG initialization failed: {e}")
            traceback.print_exc()
            rag_pipeline = None

        # Agent orchestrator (this also runs an MCP health check).
        try:
            # Surface the LLM config the orchestrator will actually use so
            # "UnsupportedProtocol" failures in production have an obvious
            # smoking gun (a missing or empty OPENAI_BASE_URL).
            effective_base_url = os.getenv("OPENAI_BASE_URL") or "<unset - defaults to api.openai.com>"
            effective_api_key_prefix = (os.getenv("OPENAI_API_KEY") or "")[:7]
            print(
                f"[lazy-init] LLM config: model={os.getenv('OPENAI_MODEL', 'deepseek-chat')!r} "
                f"base_url={effective_base_url!r} api_key_prefix={effective_api_key_prefix!r}"
            )
            print("[lazy-init] Initializing agent orchestrator...")
            orchestrator = AgentOrchestrator(
                rag_pipeline=rag_pipeline,
                mcp_server_url=mcp_server_url,
                model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
                use_mcp_protocol=True,
            )
            print("[lazy-init] Agent orchestrator initialized")
        except Exception as e:
            print(f"[lazy-init] ERROR: orchestrator init failed: {e}")
            traceback.print_exc()
            orchestrator = None

        print("[lazy-init] Heavy component initialization finished")
    except Exception as e:
        # Last-resort guard: capture the error and let the request proceed
        # so the user gets a clear 503 instead of a hung connection.
        _init_error = e
        print(f"[lazy-init] FATAL: unhandled error during init: {e}")
        traceback.print_exc()


async def initialize_heavy_components() -> None:
    """Async wrapper that runs the synchronous init in a worker thread.

    Designed to be called either from the lifespan (as a fire-and-forget
    pre-warm) or from ``ensure_initialized`` on first request. Either
    way, only one instance of the work actually runs.
    """
    if _init_event is not None and _init_event.is_set():
        return
    await asyncio.to_thread(_init_heavy_components_sync)
    if _init_event is not None:
        _init_event.set()


async def ensure_initialized(timeout: float = 120.0) -> bool:
    """Wait until heavy components are initialized, or raise on timeout.

    Returns ``True`` on success, ``False`` if initialization previously
    failed (so callers can return a clear 503 to the user). Concurrent
    callers all observe the same ``_init_event`` so the heavy work runs
    exactly once even if five requests arrive simultaneously.

    Defensive: the lifespan is the canonical place that creates the
    ``_init_event`` / ``_init_lock`` primitives, but TestClient and
    certain test fixtures can bypass the lifespan entirely. In that
    case we lazily create the primitives here so the function never
    hard-crashes on a missing attribute.
    """
    global _init_event, _init_lock, _init_error

    if _init_event is None:
        _init_event = asyncio.Event()
        _init_lock = asyncio.Lock()

    if _init_event.is_set():
        return _init_error is None and orchestrator is not None

    async with _init_lock:
        # Double-checked locking: another coroutine may have finished
        # initialization while we were waiting for the lock.
        if _init_event.is_set():
            return _init_error is None and orchestrator is not None

        try:
            await asyncio.wait_for(initialize_heavy_components(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[ensure_initialized] init exceeded {timeout}s timeout")
            return False
        return _init_error is None and orchestrator is not None


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Only does the cheap, must-happen-before-listen() work here:

    * prints env-var status,
    * creates the asyncio primitives used by ``ensure_initialized``,
    * fires a background task to pre-warm heavy components so the first
      real request typically does not pay the full init cost.

    The FastAPI ``listen()`` call returns within a few hundred ms so
    Render's health probe is satisfied immediately.
    """
    global _init_event, _init_lock

    print("=" * 60)
    print("Starting HR Policy Assistant (lazy mode)...")
    print("=" * 60)

    print(f"OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'NOT SET')}")

    # Concurrency primitives must be created on the running loop.
    _init_event = asyncio.Event()
    _init_lock = asyncio.Lock()

    # Fire-and-forget pre-warm. The task itself is held so we can cancel
    # it cleanly during shutdown, but the event loop never awaits it
    # before ``yield`` — that's the whole point.
    prewarm_task = asyncio.create_task(
        initialize_heavy_components(), name="heavy-init-prewarm"
    )

    print("[lifespan] Background pre-warm task scheduled")
    print("=" * 60)

    try:
        yield
    finally:
        # Shutdown: cancel the pre-warm if it is still running, then
        # tear down the MCP subprocess.
        if not prewarm_task.done():
            prewarm_task.cancel()
            try:
                await prewarm_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            stop_mcp_server()
        except Exception as e:
            print(f"[lifespan] WARN: stop_mcp_server raised: {e}")
        print("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="HR Policy Assistant",
    description="Agentic AI system for HR policy and operations tasks with MCP integration",
    version="1.0.0",
    lifespan=lifespan,
)

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
    """Liveness probe — returns 200 as soon as the worker is listening.

    Intentionally does not touch the MCP client or the RAG pipeline so
    Render's health check never blocks on heavy initialization.
    """
    return HealthResponse(
        status="healthy",
        app_status="running",
        initialized=_init_event is not None and _init_event.is_set() and _init_error is None,
    )


@app.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """Readiness probe — reports heavy-component status without blocking.

    Mirrors what the old ``/health`` endpoint used to do, minus the live
    MCP round-trip (which we avoid here so a stuck MCP server cannot
    make ``/ready`` itself time out — ops can still see MCP state via
    ``/mcp/status``).
    """
    initialized = (
        _init_event is not None
        and _init_event.is_set()
        and _init_error is None
        and orchestrator is not None
    )

    index_status = "not_ready"
    if rag_pipeline is not None:
        stats = rag_pipeline.get_index_stats()
        if stats.get("status") == "indexed":
            index_status = f"ready ({stats.get('chunks', 0)} chunks)"

    return ReadinessResponse(
        status="ready" if initialized else "initializing",
        initialized=initialized,
        mcp_connected=mcp_client is not None,
        index_status=index_status,
        tools_count=None,
        init_error=str(_init_error) if _init_error else None,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for HR policy questions.

    Waits for lazy initialization to complete on the very first call;
    subsequent calls hit the cached globals.
    """
    ok = await ensure_initialized()
    if not ok:
        # Either we are still warming up past the timeout, or init failed.
        if _init_error is not None:
            raise HTTPException(
                status_code=503,
                detail=f"Agent initialization failed: {_init_error}",
            )
        raise HTTPException(
            status_code=503,
            detail="Agent is still initializing. Please retry shortly.",
        )

    try:
        history = None
        if request.history:
            history = [{"role": m.role, "content": m.content} for m in request.history]

        result = await orchestrator.process_request(
            query=request.message,
            employee_id=request.employee_id,
            conversation_history=history,
        )

        return ChatResponse(
            answer=result["answer"],
            citations=result.get("citations", []),
            tool_calls=result.get("tool_calls", []),
            trace=result.get("trace", []),
            metadata=result.get("metadata", {}),
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}",
        )


@app.get("/chat/history")
async def get_history():
    """Get empty history (placeholder for session management)."""
    return {"history": []}


@app.get("/capabilities")
async def get_capabilities():
    """Get agent capabilities."""
    ok = await ensure_initialized()
    if not ok:
        return {"error": "Agent not initialized", "initializing": True}
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
            "server_url": mcp_client.server_url,
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.get("/employees")
async def list_employees():
    """List active employees for the frontend picker."""
    import json
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
    """Demo: PTO Request Guidance workflow."""
    ok = await ensure_initialized()
    if not ok:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    demo_query = "Can I take 3 days of PTO next week?"
    result = await orchestrator.process_request(
        query=demo_query,
        employee_id="EMP001",
    )
    return {
        "task": "PTO Request Guidance",
        "query": demo_query,
        "response": result,
    }


@app.get("/demo/remote-work")
async def demo_remote_work():
    """Demo: Remote Work Eligibility workflow."""
    ok = await ensure_initialized()
    if not ok:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    demo_query = "Can I work remotely from another state for 6 weeks?"
    result = await orchestrator.process_request(
        query=demo_query,
        employee_id="EMP002",
    )
    return {
        "task": "Remote Work Eligibility",
        "query": demo_query,
        "response": result,
    }


# =============================================================================
# Web UI
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the web UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    response = HTMLResponse(html_path.read_text(encoding="utf-8"))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# =============================================================================
# Policy Document Viewer
# =============================================================================

@app.get("/policy/section")
async def get_policy_section(document_id: str, section: str):
    """
    Return the chunks for a specific policy document and section heading.
    """
    import json
    from pathlib import Path

    if not document_id or not section:
        raise HTTPException(status_code=400, detail="document_id and section are required")

    safe_doc = Path(document_id).name
    vector_store_path = Path(__file__).parent.parent.parent / "data" / "vector_store" / "chunks.json"
    if not vector_store_path.exists():
        raise HTTPException(status_code=503, detail="Index not ready")

    chunks = json.loads(vector_store_path.read_text(encoding="utf-8"))

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

    if not matches:
        section_content = extract_section_from_text(full_content, section, doc_path.suffix)
    else:
        section_content = "\n\n---\n\n".join(m["content"] for m in matches[:2])

    if not section_content:
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
    """Serve a policy document for the frontend viewer modal."""
    from pathlib import Path

    policies_dir = Path(__file__).parent.parent.parent / "policies"
    safe_name = Path(filename.replace("/", "").replace("\\", "")).stem
    candidates = [
        policies_dir / f"{safe_name}.md",
        policies_dir / f"{safe_name}.txt",
        policies_dir / f"{safe_name}.html",
        policies_dir / f"{safe_name}.pdf",
    ]
    for path in candidates:
        if path.exists():
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
                    title = None
                    if reader.metadata and reader.metadata.get("/Title"):
                        title = reader.metadata.get("/Title")
                    if not title:
                        title = path.stem.replace("-", " ").replace("_", " ").title()
                    html = f"<pre style='white-space:pre-wrap;word-break:break-word;'>{content}</pre>"
                    return {
                        "filename": path.name,
                        "title": title,
                        "content": content,
                        "html": html,
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

            content = path.read_text(encoding="utf-8")
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
    html = re.sub(r'^###### (.+)$', r'<h6>\1</h6>', html, flags=re.MULTILINE)
    html = re.sub(r'^##### (.+)$', r'<h5>\1</h5>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', html, flags=re.MULTILINE)
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
    """Pull a focused snippet from the raw document by locating the section heading."""
    import re
    section_lower = section.lower().strip()
    if not section_lower:
        return ""

    lines = content.split("\n")
    is_html = suffix.lower() in (".html", ".htm")
    is_pdf = suffix.lower() == ".pdf"

    start_idx = -1
    matched_level = None
    if is_html:
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
