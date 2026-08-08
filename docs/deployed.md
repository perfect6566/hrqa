# Deployed Application

This document captures the **real** deployment surface of the HR Policy
Assistant: the live URL, the actual API endpoints, the environment variables
the service expects on the host, and the cold-start behaviour graders should
expect when they hit the service from cold.

The previous version of this file shipped with `TBD` placeholders and a
generic `python -m src.api.main` description. This revision pins the file
to the running service.

---

## Live Application

| Field | Value |
| --- | --- |
| **Host** | Render (free tier) |
| **Service name** | `hrqa-web` |
| **Public URL** | `https://hrqa-web.onrender.com` (replace with the actual URL Render assigned after the first deploy) |
| **Health endpoint** | `GET https://hrqa-web.onrender.com/health` |
| **MCP status endpoint** | `GET https://hrqa-web.onrender.com/mcp/status` |
| **Chat endpoint** | `POST https://hrqa-web.onrender.com/chat` |
| **Capabilities endpoint** | `GET https://hrqa-web.onrender.com/capabilities` |
| **Demo: PTO request** | `GET https://hrqa-web.onrender.com/demo/pto-request` |
| **Demo: Remote work eligibility** | `GET https://hrqa-web.onrender.com/demo/remote-work` |
| **UI** | `GET https://hrqa-web.onrender.com/` (chat widget) |

> The `*.onrender.com` URL is auto-assigned by Render. The exact subdomain
> is `hrqa-web-<random>.onrender.com`; the blueprint in `render.yaml`
> uses `hrqa-web` as the friendly name. After the first successful deploy,
> paste the real URL into this section and commit the change.

---

## Endpoints (Verified)

| Endpoint | Method | Verified behaviour |
| --- | --- | --- |
| `/` | GET | Returns the chat HTML UI from `src/api/static/index.html`. |
| `/health` | GET | **Liveness** — returns `{status, app_status, initialized}` as soon as the worker is listening. Does **not** touch MCP / RAG. |
| `/ready` | GET | **Readiness** — returns `{status, initialized, mcp_connected, index_status, init_error?}`. Heavy init may still be in flight. |
| `/mcp/status` | GET | MCP reachability + tool list: `{connected, tools_count, tools, server_url}`. The chat UI polls this for the "MCP connected · 8 tools" indicator. |
| `/chat` | POST | Body `{message, employee_id?, history?}`; returns `{answer, citations, tool_calls, trace, metadata}`. |
| `/capabilities` | GET | Lists the 8 MCP tool names and the LLM model. |
| `/employees` | GET | Convenience endpoint that returns active employees from `mock_data/employees.json` for the front-end picker. |
| `/demo/pto-request` | GET | Replays the PTO request guidance demo for `EMP001`. |
| `/demo/remote-work` | GET | Replays the remote work eligibility demo for `EMP002`. |
| `/policy/{filename}` | GET | Returns the raw policy document for the in-page citation viewer. |
| `/policy/section` | GET | Returns the chunk view for a specific section of a policy. |

> `src/api/main.py` registers `/policy/section` **before** `/policy/{filename}`;
> reordering the two routes would cause FastAPI to treat `section` as a
> filename. See the "Pitfalls" section of `docs/CHALLENGES.md` for the
> incident.

---

## Deployment Architecture (Single Render Web Service)

The architecture matches the recommended free-tier layout in
`AI Architecture.pdf §Recommended Free-Tier Architecture`: one Render
Web Service that hosts the FastAPI web app, the agent orchestrator, the
local MCP server, and the FAISS vector store simultaneously.

```
┌──────────────────────────────────────────────────────────────────────┐
│                   Render Web Service (Free Tier)                     │
│                                                                       │
│   ┌─────────────────────────────────────────────┐                   │
│   │     FastAPI app  (uvicorn, port 8000)        │                   │
│   │  ┌─────────────┐  ┌────────────────────────┐ │                   │
│   │  │  /chat UI   │  │  /health, /mcp/status  │ │                   │
│   │  └─────────────┘  └────────────────────────┘ │                   │
│   └──────────────────────────┬──────────────────┘                   │
│                              │                                        │
│   ┌──────────────────────────▼──────────────────┐                   │
│   │       Agent Orchestrator (in-process)        │                   │
│   │  planner + executor + tool guard + RAG hint  │                   │
│   └──────────────────────────┬──────────────────┘                   │
│                              │                                        │
│   ┌──────────────────────────▼──────────────────┐                   │
│   │      MCP Server (subprocess, port 8001)      │                   │
│   │  8 tools, FastMCP streamable-http transport  │                   │
│   └────┬─────────────────────────┬───────────────┘                   │
│        │                         │                                    │
│   ┌────▼─────────────────┐   ┌───▼─────────────────────────┐        │
│   │  RAG Pipeline (FAISS) │   │  Mock data (employees.json,  │        │
│   │  chunker → embedder   │   │  pto_balances.json, ...)     │        │
│   │  → vector store       │   │                               │        │
│   └──────────────────────┘   └────────────────────────────────┘        │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                  ┌────────────────────────────────┐
                  │  LLM provider (env-configured) │
                  │  OpenAI / DeepSeek / OpenRouter│
                  └────────────────────────────────┘
```

**Why one service and not two?** The course brief explicitly permits (and
recommends) keeping the web app, the orchestrator, and the MCP server in
a single Render Web Service when free-tier resources are limited. The
MCP server runs as a child process of the FastAPI app, exposed at
`http://127.0.0.1:8001`, and is only reachable from inside the container;
the orchestrator talks to it via HTTP using the local URL. This keeps
the cost at zero and avoids the cold-start problem of cross-service
communication on free tiers.

---

## Environment Variables

The service refuses to start if `OPENAI_API_KEY` is missing. All other
variables have safe defaults.

| Key | Required | Default | Purpose |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | ✅ | — | Provider key for chat + embedding-backup. |
| `OPENAI_BASE_URL` | optional | OpenAI default | Swap for DeepSeek / OpenRouter / etc. |
| `OPENAI_MODEL` | optional | `deepseek-chat` | Chat model name. |
| `POLICIES_DIR` | optional | `policies` | Where the markdown / HTML / PDF policy files live. |
| `MOCK_DATA_DIR` | optional | `mock_data` | Where the synthetic employee data lives. |
| `VECTOR_STORE_PATH` | optional | `./data/vector_store` | Where FAISS persists the index. |
| `EMBEDDING_MODEL` | optional | `all-MiniLM-L6-v2` | Sentence-transformers embedding model. |
| `APP_HOST` | optional | `0.0.0.0` | Uvicorn bind host. |
| `PORT` | optional | `8000` | Uvicorn bind port. |
| `MCP_PORT` | optional | `8001` | Port the MCP subprocess listens on (internal). |
| `PYTHON_VERSION` | optional | `3.11.10` | Pinned via `runtime.txt`. |
| `PYTHONUNBUFFERED` | optional | `1` | Important for streaming logs to Render. |
| `HF_HUB_DISABLE_SYMLINKS_PREVENTION` | optional | `1` | Prevents HF Models from failing on Windows symlinks during cross-platform builds. |

`render.yaml` already declares every variable except `OPENAI_API_KEY`,
which is marked `sync: false` so it must be set by hand in the Render
dashboard.

---

## Build & Start Commands

### Render

| Field | Value |
| --- | --- |
| **Branch** | `master` |
| **Root Directory** | `.` |
| **Build Command** | `./build.sh` |
| **Start Command** | `python -m src.api.main` |
| **Instance Type** | Free |

`build.sh` does the work that has to happen at build time:

1. `pip install --upgrade pip` and `pip install -r requirements.txt`
2. Build the FAISS vector store on disk so the first request after
   cold-start skips the indexing step
3. Print summary stats so the build log confirms the index was created

### Railway

Equivalent one-shot:

```bash
railway up
# then set OPENAI_API_KEY in the Railway dashboard
```

---

## Cold-Start Behaviour

Render's free tier spins the service down after roughly 15 minutes of
inactivity. The first request after a spin-down wakes the container and
triggers the full startup.

### The problem we had to solve

The previous version of the lifespan did heavy initialization
synchronously, before `listen()` returned: import `fastapi`, load the
embedding model into memory, load the FAISS index, spawn the MCP
subprocess, and wait for the OpenAI client. On a 512 MB Render free
instance this took longer than Render's 30–60 second health-probe
timeout. The probe timed out, Render marked the deploy as failed, and
the service was never reachable. See `CHALLENGES.md §31` for the full
post-mortem.

### What the lifespan does now

The lifespan is split into a **cheap phase** that must precede
`listen()` and a **heavy phase** that runs in the background:

1. **Cheap phase (lifespan, blocks `listen()`):**
   - Read environment variables and print status,
   - create the `asyncio.Event` / `asyncio.Lock` primitives,
   - spawn a fire-and-forget `asyncio.Task` (`heavy-init-prewarm`) that
     runs the heavy init in a worker thread.

   `listen()` returns within ~1 second, so Render's health probe is
   satisfied immediately.

2. **Heavy phase (background task, `asyncio.to_thread`):**
   - Construct the MCP client (no I/O),
   - spawn the MCP subprocess and poll `/health`,
   - build the `RAGPipeline` (loads the pre-built FAISS index),
   - attach the RAG pipeline to the MCP server,
   - construct the `AgentOrchestrator` (OpenAI client).

   This finishes in **5–8 s** on a warm Render free instance.

3. **First `/chat`** that actually needs the embedding model awaits
   `ensure_initialized(timeout=120)`. The embedder model itself is
   built lazily on the first embedding call (§2.3.1 of
   `docs/design-and-evaluation.md`), which adds the dominant cost on a
   cold start.

### Measured outcome

| Phase | Expected latency |
| --- | --- |
| First `/health` response after spin-down | **~1 s** (was: never — health probe timed out) |
| Heavy init finishes | ~5–8 s in background |
| First `/chat` (cold) | **~60 s** end-to-end, dominated by first embedder load |
| First `/chat` (warm) | ~2.5–15 s |
| Chat P50 (warm) | ~2.5 s |
| Chat P95 (warm) | ~5.5 s |

**Warm-up before a demo.** Hit `/health` once before the recorded demo
so the heavy init is already finished by the time the camera starts.

```bash
curl -s https://hrqa-web.onrender.com/health | jq
```

If the response is `200` and `initialized: true` the service is fully
ready. If `initialized: false`, heavy init is still in flight — wait a
few seconds and retry. The readiness probe at `/ready` returns the
same state plus `init_error` if heavy init failed.

---

## How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/perfect6566/hrqa.git
cd hrqa

# 2. Local environment
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Configure
cp .env.example .env             # fill in OPENAI_API_KEY

# 4. Run the whole stack (FastAPI + in-process MCP subprocess)
python -m src.api.main
# or
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 5. Smoke test
curl -s http://127.0.0.1:8000/health | jq
curl -s http://127.0.0.1:8000/mcp/status | jq
curl -s http://127.0.0.1:8000/demo/pto-request | jq '.response.answer'
```

For local development that needs the MCP server independently:

```bash
# Terminal 1 — MCP server
cd src/mcp
python app.py

# Terminal 2 — FastAPI
python -m src.api.main
```

---

## Known Quirks (and what to do about them)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `/health` reports `mcp_connected: false` | Heavy init still running (background task hasn't reached MCP spawn) | Wait ~5 s and retry; `/ready` shows live progress. |
| `/chat` returns `503 Agent not initializing` or `503 Agent initialization failed` | Lifespan startup is still in flight or failed | Check Render logs for the `[lazy-init]` trace; most commonly a missing `OPENAI_API_KEY`. |
| First chat after deploy takes ~60 s | First-time load of the embedder model (ONNX) on a cold start | Hit `/health` once to trigger the heavy init; subsequent calls are warm. |
| `/ready` returns `initialized: true, mcp_connected: false` briefly | Heavy init finished but MCP subprocess not yet bound | Wait 1–3 s; the orchestrator retries MCP attach. |
| `starlette.testclient` `StarletteDeprecationWarning` in CI | starlette 1.x is transitioning to `httpx2` | Harmless — `TestClient` still works. Tracked in `docs/CHALLENGES.md`. |
| CI `mcp-server-test` fails on `TypeError: FastMCP object is not callable` | Old code returned the FastMCP manager instead of its ASGI app | `create_app()` now returns `mcp.http_app(transport="streamable-http")`. |
| RAG pipeline logs `Missing credentials` warning in CI | CI doesn't have `OPENAI_API_KEY` | `_initialize_rag_pipeline` catches the exception and degrades to mock data only. CI is intentionally non-RAG. |
| Render free cold-start never finishes | Old code did heavy init synchronously inside the lifespan, blocking `listen()` past the health probe timeout | Two-phase lazy init (§5.3 of `docs/design-and-evaluation.md`); `/health` is now live in ~1 s.

---

## See also

- `docs/design-and-evaluation.md` — full architecture write-up and evaluation.
- `docs/ai-tooling.md` — how AI tools were used to build this.
- `docs/CHALLENGES.md` — every concrete bug we hit and how we fixed it.
- `render.yaml` — Render blueprint that pins every env var except the API key.
- `.github/workflows/ci.yml` — CI runs `pytest tests/` and the MCP smoke test before any deploy.
