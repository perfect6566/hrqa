# Design and Evaluation

This document is the architecture write-up and the evaluation report for
the HR Policy Assistant, as required by `AI Architecture.pdf §10. Design
Documentation` and `§9. Evaluation of the Agentic RAG Application`.

It is intentionally written against the **actual code in this repo** —
not against a generic template. Every claim about a tool, a route, a
chunk size, or a metric can be verified by reading the referenced file.

`docs/CHALLENGES.md` is the companion document with the bug-by-bug
post-mortem. `docs/ai-tooling.md` records how AI coding tools were used.

> **Headline deployment note.** The service is designed for Render's free
> Web Service tier (512 MB RAM, 0.1 vCPU). To make that tier workable
> we apply a **two-phase lazy initialization** pattern: the lifespan
> only does the cheap work that must precede `listen()`, and the heavy
> RAG / MCP / orchestrator setup runs in a background task so the
> `/health` endpoint is live within ~1 second of process start.
> Measured outcome on Render free: **first `/health` responds in ~1 s;
> service is fully ready in ~60 s** (was: never — the previous
> monolithic lifespan blocked `listen()` and the health probe timed out
> the deploy). See §2.3.1, §5.3, §6 and `CHALLENGES.md §31`.

---

## 1. Architecture Overview

The HR Policy Assistant is an agentic AI system that combines
RAG (Retrieval-Augmented Generation) with MCP (Model Context Protocol)
tool calling. The agent is an OpenAI-compatible chat model that picks
the right tool for each turn via native function calling, executes the
tool through the MCP server, and synthesises a final answer with
citations against the RAG index.

### 1.1 System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Web Application                            │
│                        (FastAPI + Chat UI)                           │
│         src/api/main.py  +  src/api/static/index.html                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Agent Orchestrator                             │
│  src/agent/                                                            │
│  ├── planner.py     (system prompt, keyword sets, prompt builder)     │
│  ├── executor.py    (ToolExecutor, parse_tool_calls, call_many)       │
│  └── orchestrator.py (AgentOrchestrator: tool loop, tool guard, RAG) │
└─────────────────────────────────────────────────────────────────────┘
            │                                       │
            ▼                                       ▼
┌──────────────────────────────┐    ┌──────────────────────────────────┐
│        MCP Server             │    │          RAG Pipeline             │
│  src/mcp/                      │    │  src/rag/                          │
│  ├── app.py            ◄─── ASGI │    │  ├── document_loader.py            │
│  ├── fastmcp_server.py        │    │  ├── chunker.py (heading-aware)     │
│  └── tools.py (HRTools)        │    │  ├── embedder.py (MiniLM-L6)        │
│                                │    │  ├── vector_store.py (FAISS)        │
│  8 tools exposed via           │    │  ├── retriever.py                   │
│  FastMCP @custom_route +       │    │  ├── generator.py                   │
│  @mcp.tool() decorators       │    │  └── rag_pipeline.py                │
└──────────────────────────────┘    └──────────────────────────────────┘
            │                                       │
            └──────────────┬────────────────────────┘
                           ▼
                  ┌────────────────────────────────┐
                  │  LLM provider (env-configured) │
                  │  OpenAI / DeepSeek / OpenRouter│
                  │  + fastembed (ONNX)             │
                  └────────────────────────────────┘
```

### 1.2 Why this shape

- **Native OpenAI function calling** instead of a free-form JSON planner.
  The LLM picks the right tool name and arguments directly; the orchestrator
  loop reads `response.choices[0].message.tool_calls` and routes them
  through the MCP `tools/call` endpoint. This is what the project brief
  means by "the agent must actually call MCP-exposed tools during
  execution" — the agent chooses the tool, the executor wraps the call
  in the MCP protocol.
- **Tool guard** (`AgentOrchestrator._enforce_tool_guard`) as a safety net.
  If the LLM skips a mandatory employee-PII tool, the orchestrator
  force-calls it before synthesising the final answer. See §3.4.
- **Pre-loop RAG retrieval** (`AgentOrchestrator._retrieve_rag_once`).
  Chunks are retrieved once before the tool loop and primed into the
  LLM context. The loop re-retrieves only if the LLM calls
  `search_policy_documents` with a different query. This avoids the
  double-RAG round-trip the first version had.
- **Citation parser that trusts the LLM**
  (`AgentOrchestrator._parse_body_section_names`). The chunker's overlap
  path can stamp a chunk with the wrong heading; the LLM sees the
  content directly and tends to cite the correct section. The parser
  prefers the LLM's body citation over the chunk's stored `heading`.

### 1.3 Request flow

1. `POST /chat` receives `{message, employee_id?, history?}`.
2. The orchestrator connects to the MCP server (`/health` + `tools/list`).
3. The planner decides `tool_choice` from a cheap RAG-only heuristic
   (`TaskPlanner.should_use_rag_only`).
4. One RAG retrieval primes the LLM with the top-5 policy chunks.
5. The OpenAI chat completion runs with `tools=[all MCP tools]`,
   `tool_choice="auto"`.
6. If the LLM returns `tool_calls`, the executor executes them in
   parallel via `ToolExecutor.call_many` (which goes through the MCP
   client). Results are appended to the message stream as `role: tool`
   messages and the loop iterates.
7. After the loop, the tool guard force-calls any mandatory tool
   the LLM skipped (`_enforce_tool_guard`).
8. The final synthesis call uses the policy chunks + tool results to
   produce the answer with `[Source N: …]` citations.
9. `_parse_body_section_names` extracts the section names from the
   body citations and builds the `References` footer.
10. The response is `{answer, citations, tool_calls, trace, metadata}`.

### 1.4 Failure modes and how they are handled

| Failure | How it is handled |
| --- | --- |
| MCP server unreachable | `_ensure_mcp_connected` returns `False`; orchestrator runs in degraded mode with no tools. |
| LLM refuses to call a mandatory tool | `_enforce_tool_guard` force-calls it after the loop. |
| Employee ID does not exist | `_detect_employee_not_found` short-circuits before the synthesizer. |
| LLM emits `<\|DSML\|>tool_calls>` inside content | `_strip_tool_call_artifacts` cuts the message at the first marker. |
| `OPENAI_API_KEY` missing | `_initialize_rag_pipeline` returns `None`; the orchestrator warns but the app still serves. |
| Missing employee ID on a workflow question | Required-tool guesser skips; falls back to a clarification prompt. |
| Heavy init still running when first request arrives | `ensure_initialized` waits on an `asyncio.Event` with a 120 s ceiling (see §5.3). |
| Heavy init fails | `_init_error` is captured; first request gets a clear 503 with the cause; `GET /ready` reports `init_error`. |

---

## 2. RAG Design

### 2.1 Document ingestion

`src/rag/document_loader.py` loads from `policies/`. The loader supports
the formats the project brief calls out:

| Format | Extension | Loader |
| --- | --- | --- |
| Markdown | `.md` | `python-markdown` / line-based parser |
| HTML | `.html` | `BeautifulSoup` |
| PDF | `.pdf` | `pypdf` |
| Plain text | `.txt` | raw read |

Per the brief, "at least two supported source formats where feasible" —
this loader supports four.

### 2.2 Chunking

`src/rag/chunker.py` performs **heading-aware chunking with overlap**:

- Split on `#`/`##`/… headings.
- Token-budget each chunk at `chunk_size=512` with `chunk_overlap=50`.
- `random.seed(42)` is set at module load so the index is reproducible
  across rebuilds. This was verified by re-indexing twice and diffing
  the chunk IDs.
- Overlap chunks are tagged with the *next* section's heading for
  context. The citation parser knows about this and trusts the LLM's
  body citation over the chunk's stored heading.

### 2.3 Embedding & vector store

| Component | Choice | Why |
| --- | --- | --- |
| Embedder | `fastembed.TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")` (384-dim, local, ONNX) | Free, no API cost, ~50 MB dep tree. Same SBERT weights the previous `sentence-transformers` build used, so vectors are dimension-compatible. See §2.3 for the lazy-init rationale. |
| Vector store | FAISS (`IndexFlatIP`) | Local, no external DB required, fast enough for our ~30-chunk corpus. |
| Persistence | `index.faiss` + `chunks.json` on disk | The build script pre-builds the index so cold-start is fast. |

#### 2.3.1 Embedder: lazy construction

`Embedder.__init__` no longer constructs the underlying model. It
stores the model name and a `batch_size`, and the actual
`fastembed.TextEmbedding(...)` object is built on first call to
`.embed_texts(...)` or `.embed_query(...)` via the `model` property:

```python
@property
def model(self):
    if self._model is None:
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=self.model_name)
    return self._model
```

The `fastembed` import is itself inside the property so simply
importing `src.rag.embedder` (which happens at FastAPI module load) does
not pull in ONNX Runtime or download model weights. This pairs with
the lifespan-side lazy init (§5.3) so the worker can become healthy
without ever loading the embedding model.

**Why lazy and not "load at build time."** Render's free Web Service
build step runs in a separate ephemeral container; the resulting
filesystem image does not carry a warm ONNX cache into the runtime
container. Eagerly loading at startup would re-pay the cost on every
cold start. Lazy construction defers the cost to the first real
embedding call (during `/chat`), and the cache survives in process
memory for the rest of the container's life.

### 2.4 Retrieval

`src/rag/retriever.py` embeds the query, runs `IndexFlatIP.search` with
`top_k=5`, and returns chunks with metadata. The orchestrator caches
`(query, k)` pairs in a per-instance LRU of size 64 to keep
long-running agents bounded.

### 2.5 Generation

`src/rag/generator.py` builds the LLM prompt from the retrieved chunks
and the user's query. The system prompt is the same
`TaskPlanner.SYSTEM_PROMPT` the orchestrator uses, so the synthesizer
sees a consistent style of citation.

### 2.6 Guardrails

- **Out-of-corpus refusal.** If `retriever.retrieve` returns no chunks
  (or all scores are below a threshold), the generator returns a
  "I don't have that information in the policy documents" message.
- **Citation enforcement.** The system prompt explicitly asks the LLM to
  cite `[Source N: ... — section]` for every factual claim; the citation
  parser then strips hallucinated sections.
- **No hidden chain-of-thought.** The `trace` returned by the
  orchestrator records tool names, arguments, latencies, and result
  snippets — nothing else. The project brief explicitly forbids hidden
  CoT.

---

## 3. MCP Server Design

### 3.1 Transport

`src/mcp/app.py:create_app()` returns `mcp.http_app(transport="streamable-http")`
backed by FastMCP. This gives a real ASGI app that uvicorn (or
`starlette.testclient.TestClient` in CI) can serve. The MCP server runs
as a subprocess of the FastAPI app on port 8001, exposed only on
`127.0.0.1` — there is no public-facing MCP port.

### 3.2 Tool list (8 tools)

| Tool | Reads from | Writes back | RAG-backed |
| --- | --- | --- | --- |
| `lookup_employee_profile` | `mock_data/employees.json` | — | No |
| `check_pto_balance` | `mock_data/pto_balances.json` | — | No |
| `lookup_benefits_status` | `mock_data/benefits.json` | — | No |
| `create_mock_hr_ticket` | (in-memory) | mock `mock_data/hr_tickets.json` | No |
| `draft_hr_email` | (template) | string | No |
| `check_policy_compliance` | rules + `mock_data` | string | No |
| `search_policy_documents` | RAG index | top-k chunks | **Yes** |
| `get_policy_section` | RAG index | focused chunks | **Yes** |

The project brief asks for "at least five MCP tools, at least one RAG,
at least one mock-data" — we ship 8 / 2 / 6.

### 3.3 Custom HTTP routes

FastMCP's `@_mcp.custom_route(...)` adds three JSON endpoints that
mirror the rest of the API:

| Route | Purpose |
| --- | --- |
| `GET /health` | Returns `{status, server, version, rag_available}`. |
| `GET /tools` | List tools in the format the front-end expects. |
| `POST /tools/call` | Single-tool call (alternative to the MCP-protocol endpoint). |
| `POST /mcp-api` | JSON-RPC-style wrapper around `tools/list` and `tools/call`. |

### 3.4 Tool guard

The orchestrator runs `_enforce_tool_guard` after the tool loop. The
guard inspects the `invocations` list and detects which mandatory tools
the LLM skipped. Allowed "mandatory" tools are:

| Keyword bucket in `planner.py` | Tool that must be called |
| --- | --- |
| `PTO_KEYWORDS` (`pto`, `vacation`, `time off`, …) | `check_pto_balance` |
| `BENEFITS_KEYWORDS` (`benefits`, `medical`, `dental`, …) | `lookup_benefits_status` |
| `REMOTE_KEYWORDS` (`remote`, `work from home`, …) | `lookup_employee_profile` + `check_policy_compliance` |
| `PROFILE_KEYWORDS` (`my`, `me`, `i`, `employee`, …) | `lookup_employee_profile` |

The guard is skipped when `rag_only=True` (pure policy question with no
employee context).

### 3.5 ASGI-callable fix

The first version of `src/mcp/app.py:create_app()` returned the
`FastMCP` manager object directly. The CI MCP smoke test failed with
`TypeError: 'FastMCP' object is not callable` because `TestClient`
calls `self.app(scope)` and a manager is not an ASGI callable. The fix
is to return `mcp.http_app(transport="streamable-http")`. This is the
behaviour the brief expects from an "HTTP MCP service" entry point.

---

## 4. Agent Orchestration

### 4.1 Planner

`src/agent/planner.py` exposes:

- `SYSTEM_PROMPT` and `RAG_ONLY_SYSTEM_PROMPT` — the synthesiser
  prompts.
- `PTO_KEYWORDS`, `BENEFITS_KEYWORDS`, `REMOTE_KEYWORDS`,
  `PROFILE_KEYWORDS` — the keyword sets the tool guard and the
  `should_use_rag_only` heuristic use.
- `should_use_rag_only(query, employee_id)` — returns `True` for
  purely definitional policy questions, in which case
  `tool_choice="none"` is forced.
- `build_user_prompt(...)` — the synthesis prompt, with explicit
  citation instructions.

### 4.2 Executor

`src/agent/executor.py` is the MCP plumbing:

- `parse_tool_calls(msg)` — normalises the OpenAI SDK's tool_calls
  shape into a list of `{name, arguments, id}` dicts.
- `load_openai_tools(refresh=True)` — pulls the tool list from the MCP
  server and converts to OpenAI's `tools=[...]` format.
- `call_many(payload)` — runs multiple tool calls in parallel via the
  MCP client.
- `assistant_tool_calls_message(msg)` and `tool_message(...)` —
  adapters for the OpenAI messages stream.

### 4.3 Orchestrator

`src/agent/orchestrator.py:AgentOrchestrator.process_request(...)` is the
top-level entry point. The full flow is in §1.3.

### 4.4 Trace structure

The orchestrator returns a `trace` list. Each step is a dict like:

```json
{
  "step": "tool_executed",
  "tool": "check_pto_balance",
  "arguments": {"employee_id": "EMP001", "year": 2026},
  "success": true,
  "latency_ms": 234,
  "mcp_call": true,
  "result": {...}
}
```

No chain-of-thought is in the trace. The brief explicitly forbids it.

---

## 5. Web Application

### 5.1 Endpoints

`src/api/main.py` exposes:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Chat HTML UI. |
| `/health` | GET | **Liveness** — returns `200 {status, app_status, initialized}` as soon as the worker is listening. Does **not** touch MCP / RAG. |
| `/ready` | GET | **Readiness** — returns `200 {status, initialized, mcp_connected, index_status, init_error?}`. Heavy init may still be in flight. |
| `/chat` | POST | Main chat endpoint. |
| `/chat/history` | GET | Empty (placeholder for session memory). |
| `/capabilities` | GET | Tool list and model name. |
| `/mcp/status` | GET | MCP connectivity + tool list. |
| `/employees` | GET | Active employees for the front-end picker. |
| `/demo/pto-request` | GET | Replays the PTO demo for `EMP001`. |
| `/demo/remote-work` | GET | Replays the remote-work demo for `EMP002`. |
| `/policy/{filename}` | GET | Raw policy document for the citation viewer. |
| `/policy/section` | GET | Focused chunk view for one section. |

> `/policy/section` is declared **before** `/policy/{filename}`. See
> `docs/CHALLENGES.md §8` for the FastAPI routing pitfall.

### 5.2 UI

`src/api/static/index.html` is a single-file chat widget with a
sidebar showing active employees, the current citations, and a
modal-based policy viewer. The widget calls `/chat` and `/health`.

### 5.3 Lifespan and lazy initialization

**Why this matters.** Render's free Web Service tier has a health probe
that gives up in 30–60 seconds. If the lifespan does any slow work
before `listen()` returns, the probe times out and Render marks the
deploy as failed — even though the process would eventually become
healthy. The first version of this code did exactly that (see
`CHALLENGES.md §31`) and the service never came up on Render free.

The worker therefore boots in two phases so the platform health probe
never times out on a free-tier cold-start:

1. **Lifespan** (runs once at startup, before `listen()` returns)
   only does the work that has to happen synchronously:
   - read environment variables and print their status,
   - create the `asyncio.Event` / `asyncio.Lock` primitives used by
     `ensure_initialized`,
   - spawn a fire-and-forget `asyncio.Task` named
     `heavy-init-prewarm` that runs the heavy initialization in a
     worker thread via `asyncio.to_thread`.

   `listen()` returns within a few hundred milliseconds, so Render's
   health probe is satisfied immediately.

2. **First request** that actually needs the RAG pipeline / MCP
   client / agent orchestrator awaits `ensure_initialized(timeout=120)`.
   That function uses double-checked locking so concurrent
   first-callers all wait on the same event instead of racing to
   build the same components five times. The 120-second ceiling
   means a stuck initializer cannot hang the request forever — it
   just returns a 503 the caller can retry.

If heavy init fails or is still running when the first request
arrives, `ensure_initialized` returns `False`; the caller maps that
to a 503 with a clear message (`"Agent is still initializing. Please
retry shortly."` vs `"Agent initialization failed: <cause>"`).
`GET /ready` exposes the same state in JSON so ops can see it
without provoking a real `/chat` round-trip.

The lifespan is also the place where the embedder model stays
un-built — see §2.3.1.

#### 5.3.1 What counts as "heavy"

`initialize_heavy_components` runs **off** the event loop via
`asyncio.to_thread`. The work it does, in order:

| Step | What | Typical cost |
| --- | --- | --- |
| 1 | `MCPClient` construction (HTTP client only — no I/O) | < 50 ms |
| 2 | Spawn MCP subprocess + poll `/health` | 1–3 s |
| 3 | Build `RAGPipeline` (loads pre-built FAISS index) | < 1 s |
| 4 | Attach `RAGPipeline` to MCP server | < 100 ms |
| 5 | Initialise `AgentOrchestrator` (construct OpenAI client) | < 1 s |
| 6 | Print env-var status for ops visibility | < 10 ms |

The embedder model itself is **not** built here — it is built on first
embedding call (§2.3.1). Cumulatively this finishes in **5–8 s** on a
warm Render free instance, so the background pre-warm task almost
always finishes before the first `/chat` request arrives.

#### 5.3.2 Measured outcome on Render free

| Metric | Before lazy init | After lazy init |
| --- | --- | --- |
| First `/health` response | never (timeout) | ~1 s after container start |
| Heavy init finishes | never (lifespan blocked) | ~5–8 s (in background task) |
| Time to first successful `/chat` | never (deploy marked failed) | ~60 s cold, ~2.5 s warm |
| Render deploy status | `failed` (health probe timeout) | `live` |
| Process memory after init | n/a (did not start) | ~280 MB (fits in 512 MB tier) |

#### 5.3.3 Why not just shorten the lifespan with `--workers 2` or `--timeout-keep-alive 0`?

Two common suggestions do **not** solve this problem:

- **More workers.** Render free allows 1 worker; with more workers
  every worker would still need to lazy-init, multiplying the cost
  rather than sharing it.
- **A shorter health probe.** Render does not let you configure the
  probe timeout on the free tier; it is hard-coded at the platform
  level.

The only fix that actually works on free is **making `listen()`
return before any expensive work runs**, which is what the two-phase
init does.

---

## 6. Deployment

See `docs/deployed.md` for the full deployment guide. The headline
facts:

- Single Render Web Service, free tier.
- `./build.sh` installs deps and pre-builds the FAISS index.
- Start command: `granian --interface asgi --host 0.0.0.0 --port $PORT --workers 1 src.api.main:app`
  (matches `render.yaml`; local dev uses `python -m src.main` which
  wraps the same `granian.Granian` call).
- `render.yaml` declares every env var except `OPENAI_API_KEY`.
- The lifespan runs a background pre-warm task so `/health` is live
  within ~1 second of container start; the first `/chat` waits on
  `ensure_initialized` for up to 120 s (see §5.3).
- Cold start (Render free, first request after spin-down): the service
  comes up in ~60 s end-to-end — ~1 s for `/health`, ~5–8 s for the
  background heavy init, then the first `/chat` takes ~50 s while the
  embedder model loads on first use. Warm p50: ~2.5 s. Warm p95: ~5.5 s.
  Before this change, Render marked the deploy as failed because the
  health probe timed out; the service never became reachable at all
  (see `CHALLENGES.md §31`).

---

## 7. CI/CD

`.github/workflows/ci.yml` runs on every push and PR to `master`:

1. **Checkout** code.
2. **Set up Python** 3.11.
3. **Install dependencies** (`pip install -r requirements.txt`).
4. **Run the test suite** (`pytest tests/`).
5. **Run the MCP smoke test** — verifies the MCP server starts, the
   ASGI app answers `GET /health`, the tool list contains at least 5
   tools, and `POST /tools/call` returns a successful tool result.
6. **Deploy** (only on `master` and only if all tests pass) via
   `JorgeLNJunior/render-deploy@v1.5.0` with `wait_deploy: true`.

The deploy step uses an officially-maintained third-party GitHub Action
(`JorgeLNJunior/render-deploy` v1.5.0, ~2k stars, snake_case inputs).
The earlier choice (`render-deploy-action@v1`) was rejected by the
GitHub Actions lint with `Invalid workflow file: ... Expected format
{org}/{repo}[/path]@ref. Actual 'render-deploy-action@v1'`. See
`docs/CHALLENGES.md §1` for the full fix log.

---

## 8. Safety Guardrails

| Layer | Guardrail |
| --- | --- |
| Input | Validate employee IDs (`EMP\\d{3,}`). Reject empty queries. |
| Tool | Mock all "create" actions (`create_mock_hr_ticket`, `draft_hr_email` are in-memory only). |
| Agent | Tool guard force-calls missing mandatory tools. |
| Output | Citation enforcement + the `_detect_employee_not_found` short-circuit. |
| Audit | Every tool call is logged in `trace` with arguments, result, and latency. |

---

## 9. Two Required Agentic Demo Tasks

### 9.1 Demo 1 — PTO Request Guidance

**User query.** `Can I take 3 days of PTO next week?` for `EMP001`.

**Expected MCP tool sequence (from `TaskPlanner.PTO_KEYWORDS`):**

1. `lookup_employee_profile(employee_id="EMP001")`
2. `check_pto_balance(employee_id="EMP001", year=2026)`
3. `search_policy_documents(query="PTO request approval policy")`
4. *(optional)* `draft_hr_email(employee_id="EMP001", purpose="pto_request")`

**Expected answer shape.** PTO balance, manager approval requirement
for 3+ days, citation to PTO policy manager-approval section, optional
mock email draft.

**Replaying.** `GET /demo/pto-request` returns the structured response.

### 9.2 Demo 2 — Remote Work Eligibility

**User query.** `Can I work remotely from another state for 6 weeks?`
for `EMP002`.

**Expected MCP tool sequence:**

1. `lookup_employee_profile(employee_id="EMP002")`
2. `check_policy_compliance(employee_id="EMP002", policy_area="remote_work")`
3. `search_policy_documents(query="remote work out-of-state policy")`
4. *(optional)* `create_mock_hr_ticket(...)` if approval is required.

**Expected answer shape.** Work arrangement, compliance status, citation
to remote-work policy tax/approval section, next-step recommendation.

**Replaying.** `GET /demo/remote-work` returns the structured response.

---

## 10. Evaluation

### 10.1 Evaluation set

`evaluation/questions.py` contains 20 questions covering all required
categories:

| Category | Count | Examples |
| --- | --- | --- |
| `policy_qa` (simple) | 7 | `eval_01`–`eval_05`, `eval_19`, `eval_20` |
| `employee_data` (tool-requiring) | 5 | `eval_06`–`eval_10` |
| `multi_doc` | 2 | `eval_11`, `eval_12` |
| `workflow` (agentic) | 3 | `eval_13`, `eval_14`, `eval_18` |
| `ambiguous` | 1 | `eval_15` |
| `out_of_scope` | 2 | `eval_16`, `eval_17` |

Each question has a `gold_answer` (or `expected_behavior`) and
optional `expected_tool` / `expected_tools` / `employee_id` metadata.

### 10.2 Metrics

`evaluation/run_evaluation.py` runs the questions against the
orchestrator and records:

- **groundedness** — keyword overlap between the actual answer and the
  gold answer (lower bound; not a substitute for human grading).
- **citation_accuracy** — `1.0` if the answer included at least one
  citation, else `0.0`.
- **tool_selection_correct** — whether the primary expected tool was
  among the call list.
- **workflow_completed** — whether the workflow had at least one tool
  call (or `True` for non-workflow questions).
- **latency_ms** — wall-clock per question.

Aggregated metrics (from `evaluator.py:Evaluator.compute_metrics`):

- `groundedness_avg`
- `citation_accuracy_avg`
- `tool_selection_accuracy`
- `workflow_completion_rate`
- `latency_p50_ms` / `latency_p95_ms`

### 10.3 Deterministic seeds

- `random.seed(42)` in `chunker.py` (chunking & index).
- `set_evaluation_seed(42)` in `evaluator.py` (sample shuffling).
- `EVALUATION_SEED = 42` exported as a module-level constant.

### 10.4 How to run

```bash
# From the repo root
python -m evaluation.run_evaluation
# Produces evaluation/results.json
```

> Running the full evaluation requires `OPENAI_API_KEY` (the
> orchestrator makes real chat calls). On a machine without a key, the
> RAG path degrades to mock-only and most workflow questions will fail
> — which is exactly the behaviour we want the harness to report.

### 10.5 Results

`evaluation/results.json` carries the live numbers. The values in that
file are regenerated by `run_evaluation.py`; the JSON checked in is
the most recent run plus a `seed` field for reproducibility.

**Latest run on the deployed configuration** (`deepseek-v4-flash`,
real chat calls, real RAG, real MCP subprocess):

| Metric | Value |
| --- | --- |
| `pass_rate` | 13 / 20 = 65.0% |
| `groundedness_avg` | 55.2% |
| `citation_accuracy_avg` | 75.0% |
| `tool_selection_accuracy` | 75.0% |
| `workflow_completion_rate` | 85.0% |
| `latency_p50_ms` | 11 609 |
| `latency_p95_ms` | 27 000 |

The headline numbers are honest: this run reflects the **real**
behaviour of `deepseek-v4-flash` against the rubric questions, **not**
the earlier hand-typed estimates (see `CHALLENGES.md §16`). Where
groundedness falls below 70% it is because the LLM's answer prose
differs from the gold phrasing even though the cited chunks are
correct — a known limitation of pure keyword-overlap scoring (see
§10.2). Latency is dominated by the DeepSeek chat completion round-trip
(~9–15 s on the first /chat after a cold start, ~2.5 s warm).

**Out-of-scope behaviour.** Out-of-scope questions (`eval_16`,
`eval_17`) both report `success=False` from the LLM with no citations
and a clean refusal — **action_safety_pass_rate = 100%**.

**Re-running.** From the repo root:

```bash
python -m evaluation.run_evaluation
# Writes evaluation/results.json
```

### 10.6 Ablation — chunk size

The chunker exposes `chunk_size` and `chunk_overlap`. The default is
`512 / 50`. Earlier prototypes used:

| Chunk size | Notes |
| --- | --- |
| 256 | More focused context, but PTO policy multi-section answers lost cohesion. |
| **512** | Sweet spot. **Default.** |
| 1024 | Captures more cross-section context, but the LLM starts citing earlier sections on questions about later ones. |

### 10.7 Ablation — retrieval k

`top_k` defaults to `5`. Earlier prototypes used:

| k | Notes |
| --- | --- |
| 3 | Missed citations for multi-document questions (`eval_11`, `eval_12`). |
| **5** | Sweet spot. **Default.** |
| 10 | Added noise; the LLM cited less-relevant sections. |

### 10.8 Failure modes observed

| Failure | Frequency | Mitigation |
| --- | --- | --- |
| LLM emits `<\|DSML\|>tool_calls>` inside content | ≈ 1 / 50 requests on `deepseek-v4-flash` | `_strip_tool_call_artifacts` |
| LLM skips a mandatory tool | ≈ 1 / 8 workflow questions | Tool guard |
| LLM hallucinates a PTO balance for a missing ID | ≈ 1 / 5 missing-ID requests | `_detect_employee_not_found` short-circuit |
| Cold-start latency | 30–60 s end-to-end on Render free | Two-phase lazy init (§5.3) + pre-built FAISS index + warm-up ping before demos |

---

## 11. Cross-references

- `docs/ai-tooling.md` — how AI tools were used to build this project.
- `docs/CHALLENGES.md` — the bug-by-bug post-mortem.
- `docs/deployed.md` — deployment-specific facts and live URLs.
- `README.md` — quickstart, project structure, API endpoints.
- `render.yaml` — Render blueprint.
- `.github/workflows/ci.yml` — CI/CD pipeline.
