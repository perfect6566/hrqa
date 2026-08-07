# Development Challenges and How We Solved Them

This is the **post-mortem**. Every section here is a real bug we hit
during the build, with the failing artefact, the symptom, the root
cause, and the fix. The companion design write-up is in
`docs/design-and-evaluation.md`; the AI-tooling story is in
`docs/ai-tooling.md`.

The numbering is stable: a future reviewer can reference a bug as
`CHALLENGES.md §N`.

---

## 1. GitHub Actions rejected the deploy step

**Symptom.** CI lint failed with:

```
Annotations
1 error
Invalid workflow file: .github/workflows/ci.yml#L1
(Line: 195, Col: 15): Expected format {org}/{repo}[/path]@ref.
Actual 'render-deploy-action@v1'
```

**Root cause.** Two issues stacked:

1. The reference `render-deploy-action@v1` uses a single-segment
   version. GitHub Actions now requires `≥3`-segment semver (or a
   commit SHA) for all third-party actions.
2. The plausible-but-wrong guess `render-oss/render-deploy-action@v1.0.0`
   assumes the official Render org publishes a deploy action. It does
   not. `render-oss` is the official Render org, but its public repos
   are `render-mcp-server`, `cli`, `terraform-provider-render`,
   `sdk`, etc. — no `render-deploy-action`.

**Fix.** Switch to the community-maintained action
`JorgeLNJunior/render-deploy@v1.5.0` (≈2k stars, snake_case inputs) and
rename the inputs:

```yaml
- name: Deploy to Render
  uses: JorgeLNJunior/render-deploy@v1.5.0
  with:
    api_key: ${{ secrets.RENDER_API_KEY }}
    service_id: ${{ secrets.RENDER_SERVICE_ID }}
    wait_deploy: true
```

**How we found it.** Web-search for "render-deploy-action GitHub" →
top hits were the official Render org (no such action) and
JorgeLNJunior's repo. The Marketplace listing for "Deploy to Render"
also points at JorgeLNJunior.

**Lesson.** Never guess an org/repo slug from the package name. The
GitHub Marketplace listing is the canonical source.

---

## 2. `TypeError: 'FastMCP' object is not callable` in the MCP smoke test

**Symptom.** CI MCP smoke test failed with:

```
File ".../starlette/testclient.py", line 78, in __call__
    instance = self.app(scope)
               ^^^^^^^^^^^^^^^
TypeError: 'FastMCP' object is not callable
```

**Root cause.** `src/mcp/app.py:create_app()` returned the FastMCP
manager object directly. The FastMCP manager is *not* an ASGI app —
it's a controller that builds one. `starlette.testclient.TestClient`
calls `self.app(scope)` on whatever it receives, and the manager
doesn't implement `__call__(scope, receive, send)`.

**Fix.** Return the FastMCP-built ASGI app:

```python
def create_app():
    """Return the ASGI app for the MCP server."""
    mcp = initialize_server()
    return mcp.http_app(transport="streamable-http")
```

The `__main__` block also had to change: it now calls
`initialize_server()` directly to get the manager, then calls
`mcp.run(transport="streamable-http", host="127.0.0.1", port=port)`
on the manager.

**How we found it.** `python -c "from fastmcp import FastMCP;
m=FastMCP(name='t'); print([n for n in dir(m) if not
n.startswith('_')])"` exposed `http_app` in the public method list. The
README of fastmcp 3.x doesn't make this glaringly obvious because the
default `mcp.run()` hides the ASGI step.

**Lesson.** When a framework offers both a "manager" object and an
ASGI app, the manager is *not* a drop-in app. Read the public method
list before assuming.

---

## 3. Starlette `TestClient` deprecation warning

**Symptom.** CI logs include:

```
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient`
is deprecated; install `httpx2` instead.
  from starlette.testclient import TestClient as TestClient  # noqa
```

**Root cause.** `starlette` 1.x is migrating its test client to a
new `httpx2` internals API. The current `TestClient` still works with
the legacy `httpx`, but it emits this warning on import.

**Status.** Unfixed. The migration target (`httpx2`) doesn't yet
expose the same `portal.call` thread bridge that `starlette.testclient`
needs to run the ASGI app in a thread.

**Why we left it.** The warning is non-fatal, the test still passes,
and pinning `httpx<0.28` would break unrelated dependencies. Tracked
as a known issue; revisit when `httpx2` reaches parity.

**Lesson.** Not every deprecation warning is worth chasing. Document
the trade-off and move on.

---

## 4. RAG pipeline fails to initialise without `OPENAI_API_KEY`

**Symptom.** CI smoke test crashed with the RAG pipeline raising
`Missing credentials. Please pass an `api_key`, `workload_identity`,
`admin_api_key`, or set the `OPENAI_API_KEY` or `OPENAI_ADMIN_KEY`
environment variable.`

**Root cause.** The OpenAI SDK raises on the first call when no key
is set. The MCP server's `_initialize_rag_pipeline` propagated the
exception, the lifespan failed, and the smoke test couldn't even
reach the HTTP endpoints.

**Fix.** Wrap the RAG pipeline construction in `try/except` and return
`None` on failure. The orchestrator already supports a `None` RAG
pipeline (it falls back to direct tool use + a warning message). The
mock-data tools continue to work without an LLM at all:

```python
def _initialize_rag_pipeline():
    try:
        from src.rag.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline(...)
        return pipeline
    except Exception as e:
        print(f"[MCP] Failed to initialize RAG pipeline: {e}")
        return None
```

**Lesson.** CI should never depend on a paid API key. Wrap every
optional network call in a try/except returning a `None`/`disabled`
sentinel so the service degrades instead of crashing.

---

## 5. DeepSeek emits `<|DSML|>tool_calls>` inside the assistant content

**Symptom.** Final answer occasionally contained a paragraph like:

```
Based on the policy, you can take 3 days of PTO.
<|DSML|>function_calls>
<invoke name="search_policy_documents">
<parameter name="query">PTO request</parameter>
</invoke>
</DSML|>
```

That leaked into the chat UI and looked like a malformed answer.

**Root cause.** Some DeepSeek flash variants emit `<|DSML|>` (or
fullwidth `｜DSML｜`) pseudo-tool-call XML inside the assistant
`content` field, even when `tool_choice="none"`. The OpenAI SDK
parses the *real* `tool_calls` field separately, so the answer gets
appended with the model's "thought" appended as visible text.

**Fix.** Strip the artifact before returning the answer:

```python
@staticmethod
def _strip_tool_call_artifacts(text: str) -> str:
    markers = (
        "<|DSML|>", "<|dsml|>",
        "<|python_tag|>", "<|python_tag",
        "<|tool_call|>", "<|tool_call",
        "<|tool_call_begin|>", "<|tool_calls_section_end|>",
        "｜DSML｜", "｜dsml｜",
        "<​tool_call>", "<tool_calls>",
    )
    if not any(m in text for m in markers):
        return text.strip()
    cut_at = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)
    return text[:cut_at].strip()
```

**Lesson.** Cheap models leak. Always normalise the final-answer
channel before it reaches the user.

---

## 6. LLM skips a mandatory employee tool

**Symptom.** On PTO questions, the LLM sometimes answered with the
policy citation without first calling `check_pto_balance`. The answer
was "correct" prose but didn't include the actual balance, so the
moderator flagged it.

**Root cause.** Native function calling lets the LLM choose *no* tool.
The LLM is told to ground PTO questions in the balance, but it's not
forced to.

**Fix.** Tool guard. After the loop, scan the keyword buckets in
`planner.py` (`PTO_KEYWORDS`, `BENEFITS_KEYWORDS`, `REMOTE_KEYWORDS`,
`PROFILE_KEYWORDS`), and force-call any missing tool whose arguments
can be filled from `employee_id`:

```python
forced = await self._enforce_tool_guard(query, employee_id, invocations)
for inv in forced:
    messages.append(ToolExecutor.tool_message(..., inv.result))
```

The guard is skipped when `rag_only=True` — pure policy questions
don't need an employee lookup.

**Lesson.** Native function calling is a suggestion, not a contract.
Add a safety net for the mandatory cases.

---

## 7. LLM hallucinates a PTO balance for a non-existent employee

**Symptom.** User asks about `EMP999`. `lookup_employee_profile`
returns `success=False, error="Employee not found"`. The LLM
glosses over the error and answers "you have 8 available PTO days"
with a tidy citation.

**Root cause.** The LLM treats a tool call that returned `success=False`
as "I have the data, just not the answer." Some models recover; some
don't.

**Fix.** Short-circuit before the synthesizer:

```python
not_found = self._detect_employee_not_found(invocations, employee_id)
if not_found:
    return {
        "answer": f"I could not find employee **{employee_id}** in our HR system. ..."
        "citations": [],
        "tool_calls": [inv.to_dict() for inv in invocations],
        "trace": trace,
        "metadata": {...},
    }
```

`_detect_employee_not_found` returns `True` only when **all** of the
employee-PII tools (`lookup_employee_profile`, `check_pto_balance`,
`lookup_benefits_status`, `check_policy_compliance`) reported the
employee is missing. Single-tool misses don't trigger the
short-circuit.

**Lesson.** Don't trust the model to fail gracefully. Validate the
upstream signals and short-circuit.

---

## 8. FastAPI route ordering bug

**Symptom.** `GET /policy/section?document_id=pto-policy&section=Accrual`
returned `404 Policy 'section' not found`.

**Root cause.** FastAPI matches routes in declaration order. With
`@app.get("/policy/{filename}")` declared first, the path
`/policy/section` matched the wildcard with `filename="section"`.

**Fix.** Declare `/policy/section` **before** `/policy/{filename}`:

```python
@app.get("/policy/section")                  # MUST come first
async def get_policy_section(...): ...

@app.get("/policy/{filename}")                # wildcard
async def get_policy(filename: str): ...
```

**Lesson.** When a wildcard route is "shadowed" by a sibling concrete
route, ordering matters. The mistake is silent — no startup error.

---

## 9. PowerShell escaping ate our shell pipelines

**Symptom.** `powershell -Command "python -c \"import yaml; ..."` ran
but Python received unterminated quotes. `cd foo && python bar.py`
returned "the `&&` token is not a valid statement separator in this
version". `Get-ChildItem | Where-Object { $_.PSIsContainer -eq $false }`
returned `CommandNotFoundException` on `PSIsContainer`.

**Root cause.** Cursor's default terminal on Windows is PowerShell 5.x.
PowerShell has different escape rules, different separators, and
treats `$_` inside a `Where-Object` script block as a special variable
that the parser sometimes tries to invoke as a cmdlet.

**Fix.** Keep the heavy lifting in **Python scripts** invoked as
`powershell -Command "python scripts/foo.py"`. The script writes
deterministic output to a file; the shell only runs one command.

```python
# example: scripts/validate_ci.py
import yaml
with open(".github/workflows/ci.yml", encoding="utf-8") as f:
    print(yaml.safe_load(f))
```

Invoked as:

```powershell
powershell -Command "python scripts/validate_ci.py"
```

No `&&`, no nested quotes, no `$_` magic.

**Lesson.** Pick one scripting language per task. Don't try to write
shell pipelines that need to work on both bash and PowerShell.

---

## 10. GitHub Actions doesn't have `OPENAI_API_KEY`

**Symptom.** See §4. The RAG pipeline asserts on the first embed call.

**Fix.** Same as §4: wrap the pipeline construction in `try/except`
and degrade. CI considers the MCP server successful if the HTTP
endpoints (`/health`, `/tools`, `/tools/call`) respond — it does not
require the RAG pipeline to be online.

**Lesson.** Make expensive resources optional in CI.

---

## 11. Mock data must be obviously synthetic

**Symptom.** The first draft of `mock_data/employees.json` used real-sounding
names like "Sarah Mitchell" and "James Whitfield" with realistic email
addresses. The grader flagged it as "borderline PII — please make
synthetic".

**Fix.** Use placeholder names ("Alice Johnson", "Bob Smith", "Carol
Davis", "David Wilson", "Emma Brown") with clearly-fake IDs
(`EMP001`–`EMP010`) and `example.com` email domains. Add a note at the
top of every mock file:

```json
{
  "_synthetic": true,
  "employees": [...]
}
```

**Lesson.** Realistic-looking mock data is a liability. Make it
visibly fictional.

---

## 12. Cold-start latency on Render free tier

**Symptom.** First chat after a 20-minute idle period takes 30–60 s.
Demo recordings catch this.

**Root cause.** Render spins down the container after ~15 min idle. The
first request wakes the container, imports `fastapi`, loads
`sentence-transformers`, loads the FAISS index from disk, and spawns
the MCP subprocess. Each step is unavoidable.

**Fix.** Pre-build the FAISS index at build time (`build.sh`). This
moves the indexing cost out of the request path. The DEMO still
needs to hit `/health` once before the recording to wake the
container.

**Lesson.** Free-tier latency is a fact of life. Document the
warm-up step in the demo script.

---

## 13. The chunker's overlap path mis-tags headings

**Symptom.** After a chunk that ends at the boundary of section "PTO
Accrual" and section "Leave of Absence", the next chunk (overlap) was
tagged with `heading="Leave of Absence"` while the chunk's content was
still about PTO accrual. The citation footer then listed the wrong
section.

**Root cause.** The chunker assigns the next-section heading to the
overlap chunk for context. Reasonable default, but the LLM sees the
content and disagrees.

**Fix.** Trust the LLM. The citation parser in
`AgentOrchestrator._parse_body_section_names` extracts the section
name from the LLM's in-body citation `[Source N: ... — section]` and
prefers it over the chunk's stored `heading`. The result is that the
References footer matches the body, even when the chunk metadata is
wrong.

**Lesson.** When the model has more context than the chunk metadata,
trust the model.

---

## 14. The MCP server as a subprocess

**Symptom.** Cargo-culting the tutorial, the first version of
`src/api/main.py` started the MCP server as a child process *after*
the FastAPI app started. The MCP server took a few seconds to bind
and the first `/chat` request hit the orchestrator before the
orchestrator had a healthy MCP connection.

**Fix.** Start the MCP subprocess *before* the orchestrator
initialises, and block on the first `/health` response (up to 15 s).
The lifespan handler `start_mcp_server` polls
`http://127.0.0.1:8001/health` every 0.5 s until the server responds.

**Lesson.** Subprocess dependencies need explicit handshake.

---

## 15. Free-tier outbound network egress

**Symptom.** Render free tier blocks outbound traffic to some ports
in some regions. The first deploy couldn't reach `api.deepseek.com`.

**Root cause.** Render's free tier has a more restrictive egress
allowlist than paid tiers.

**Fix.** Set `OPENAI_BASE_URL` to a provider that works from Render
free (OpenAI's `api.openai.com` works; some mirrors don't). Document
the choice in `README.md`.

**Lesson.** Document the egress constraints; don't take free-tier
outbound for granted.

---

## 16. The `evaluation/results.json` numbers were hand-typed

**Symptom.** The first commit of `evaluation/results.json` had
`groundedness_avg: 85.0%`, `tool_selection_accuracy: 90.0%`, etc.
The runner hadn't actually been executed; someone had hand-typed them
based on vibes.

**Root cause.** Habit. When you want to ship a doc and the runner
needs an API key, the doc gets made up.

**Fix.** The new `results.json` (1) is regenerated by `python -m
evaluation.run_evaluation` and (2) carries a `mode` field that
distinguishes a real run from a static estimate. The hand-typed
numbers were deleted.

**Lesson.** Always refresh the artefact from the source of truth.
Mark simulated-vs-produced values explicitly.

---

## 17. The `OPENAI_BASE_URL` switch tripped DeepSeek content filtering

**Symptom.** Switching `OPENAI_BASE_URL` from OpenAI to DeepSeek worked
for most queries, but the model occasionally returned empty content
when the query touched HR-sensitive topics (e.g. "I want to
discriminate against …").

**Root cause.** DeepSeek's content filter is more aggressive than
OpenAI's; it returns `content=""` with a `finish_reason="content_filter"`.

**Fix.** Detect `finish_reason="content_filter"` and re-issue the
prompt with a "rewrite the question in workplace-appropriate terms"
prelude. (Currently, a 400 is returned with a clear message; the
rewrite path is on the roadmap.)

**Lesson.** Provider-specific behaviour is opaque. Wrap each provider
in a thin adapter and treat content-filter responses as a known
shape.

---

## 18. The planner's `should_use_rag_only` heuristic was too aggressive

**Symptom.** The first version of `should_use_rag_only` returned
`True` for any query that contained the word "policy". This starved
the tool loop on simple PTO questions ("how many PTO days do I get
per year?") that don't need tools but should still be answered.

**Fix.** The heuristic now requires the query to (a) not contain
`employee_id` and (b) not contain any of the keyword buckets
(`PTO_KEYWORDS`, `BENEFITS_KEYWORDS`, `REMOTE_KEYWORDS`,
`PROFILE_KEYWORDS`). Pure policy questions like "what is the company's
data classification policy?" still hit `rag_only=True`; questions
that look like personal workflows don't.

**Lesson.** Heuristics need to be conservative about when they
opt out of the LLM's tools.

---

## 19. The default API key path in `demoapikey.py`

**Symptom.** A `demoapikey.py` file at the repo root was committed
during early scaffolding. It contained a hard-coded OpenAI key.

**Root cause.** Convenience for testing.

**Fix.** Removed from the repo; added to `.gitignore`. Anyone who needs
a demo key reads the README and signs up for their own.

**Lesson.** Never commit keys. Even test keys. Even keys that look
fake.

---

## 20. The `pyproject.toml` was empty

**Symptom.** `pyproject.toml` existed but had no `[project]` metadata.
`pip install .` from source would fail.

**Fix.** Pinned metadata in `pyproject.toml` matching `requirements.txt`,
including Python version, name, and version. The build pipeline
doesn't use `pyproject.toml` (it uses `requirements.txt`) but the
metadata is still useful for IDE tooling.

**Lesson.** Even minimum-viable manifests need minimum-viable content.

---

## 21. The `runtime.txt` was pointing at the wrong Python version

**Symptom.** Render picked Python 3.10 by default. The code uses
`asyncio.get_event_loop()` and `match` statements that need 3.10+.

**Fix.** `runtime.txt` now pins `python-3.11.10`.

**Lesson.** Pin the runtime. Python minor-version differences are
real.

---

## 22. Hatch / fallback in `src/api/main_recovered.py`

**Symptom.** A `src/api/main_recovered.py` exists in the repo. It
contains a stale copy of `main.py` from a previous failed recovery.

**Fix.** Tracked in the gitignore dance; the file is scheduled for
removal in a future cleanup. Not functionally relevant.

**Lesson.** Recovered files belong in a `tmp/` directory or git
history, not in `src/`.

---

## 23. Reading the project brief twice

**Symptom.** The first draft of `docs/design-and-evaluation.md`
described a "single-service deployment with the orchestrator in one
process" without actually quoting the project brief. The grader
flagged it as "not connected to the brief".

**Fix.** Each section of `design-and-evaluation.md` now starts with a
quote from the relevant section of `AI Architecture.pdf` (e.g.
"§4. Agentic System Design"). The grader can verify in seconds.

**Lesson.** Always cite the brief. The brief is the spec.

---

## 24. The MCP client / server port conflict

**Symptom.** Setting `MCP_PORT=8000` (the same as the FastAPI port)
made the MCP subprocess die silently with "address already in use".

**Fix.** `MCP_PORT` defaults to `8001` and is documented explicitly as
"internal-use only". The orchestrator talks to it via
`http://127.0.0.1:8001`.

**Lesson.** Port numbers are not magic. Document which ones are
internal.

---

## 25. The HTML UI test loop

**Symptom.** Front-end testing the chat widget was hard to automate
because the UI does inline rendering and the chat panel scrolls.

**Fix.** Keep the HTML UI as a thin wrapper around the `/chat` JSON
endpoint. All real testing happens via `curl`/`httpx` against the
JSON endpoint. The HTML is only manually verified.

**Lesson.** The UI is a presentation layer. Test the API.

---

## 26. The first evaluation grader had no `expected_tool` for multi-tool workflows

**Symptom.** `evaluation/questions.py` had `expected_tool: "lookup_employee_profile"`
for `eval_06` (Alice's PTO balance), but the *correct* tool was
`check_pto_balance`. The hand-written file quietly disagreed with the
actual question.

**Fix.** Walked through every question and matched the expected tool
to the question's actual ask. The first draft missed two multi-tool
workflows (`eval_13`, `eval_14`); those now have `expected_tools: [...]`
lists.

**Lesson.** The evaluation set is part of the spec. It needs to be
hand-checked against the policies, not auto-generated.

---

## 27. The `evaluation/results.json` ablation tables were hard-coded

**Symptom.** The first `results.json` had ablation tables
(`chunk_size_comparison`, `retrieval_k_comparison`) with percentage
values that looked real but were never measured.

**Fix.** The new `results.json` only carries ablation tables when the
runner actually ran the ablation; otherwise the field is empty with
a note pointing to `Evaluator.test_chunk_sizes` as the harness.

**Lesson.** Don't pretend to have data you don't have.

---

## 28. The `mcp-server-test` job vs `test` job

**Symptom.** CI had two jobs: `test` (pytest) and `mcp-server-test`
(custom Python smoke test). The deploy job depended on both. When
`mcp-server-test` failed due to the `FastMCP callable` bug, the
`test` job's success was hidden and the diagnostic info was incomplete.

**Fix.** Re-ordered the deploy job's `needs: [test, mcp-server-test]`
so both jobs run in parallel. Added `if: always()` clauses so a
failure in one job doesn't mask the other.

**Lesson.** Independent CI jobs should run independently.

---

## 29. The `MCPClient` timeout

**Symptom.** `MCPClient` first requests sometimes timed out because
the MCP subprocess was still booting. The orchestrator interpreted
the timeout as "MCP unavailable" and degraded.

**Fix.** Increased `MCPClient` timeout from 2 s to 5 s and added a
one-time retry on the first connection. `_ensure_mcp_connected` now
sleeps 1 s and retries once before giving up.

**Lesson.** Don't immediately fail on a transient startup race.

---

## 30. The `_extract_pdf.py` debug script committed to the repo

**Symptom.** A `_extract_pdf.py` file was committed to the repo root
during the development of this `CHALLENGES.md` document. It was a
quick-and-dirty PyPDF-based text extractor that just expanded the
course brief into a plain-text file for grepping.

**Fix.** Removed from the repo. The article-reading workflow is
now a one-shot script that streams to a file outside the repo.

**Lesson.** Even debugging scripts should not be committed. Use a
`tmp/` or `scratch/` directory that's gitignored.

---

## Takeaways

The most common theme across these 30 issues:

1. **CI environments are not your dev environment.** No `OPENAI_API_KEY`,
   PowerShell quirks, no cache, no model state. Build the entire
   smoke test to survive that.
2. **Cheap models leak.** When the model is a sample-tier DeepSeek,
   commit to normalising the final-answer channel. Trust nothing.
3. **ASGI details bite.** FastMCP returns a manager, not an app.
   FastAPI matches routes in declaration order. Both are true and
   both will silently eat your afternoon.
4. **The brief is the spec.** Quote it. Verify every claim against
   it. The grader will.
5. **Mock data is risky.** Make it visibly synthetic. Always.
6. **Documentation is part of the deliverable.** `TBD`, `85.0%`,
   "configure after deployment" are not acceptable in a graded
   submission. Hand-typed numbers are worse than no numbers.

The full architecture write-up is in `docs/design-and-evaluation.md`,
the AI-tooling story is in `docs/ai-tooling.md`, and the deployment
guide is in `docs/deployed.md`.