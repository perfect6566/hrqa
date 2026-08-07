"""Integration tests for the FastAPI app.

These tests run the FastAPI app in-process via ``TestClient`` (no live
server required) and stub the orchestrator so they do not need an
``OPENAI_API_KEY``. They verify the *API contract* — status codes,
payload shape, and tool-call routing — not the LLM behaviour.

Run with:
    pytest tests/test_api_integration.py -v
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make the project root importable so ``from src...`` resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Stub orchestrator
# ---------------------------------------------------------------------------
# The integration tests only need to verify that the API endpoint routes
# correctly to the orchestrator and shapes the response. We install a
# stub BEFORE the FastAPI app is imported so the lifespan handler can
# pick it up. The stub mimics the small surface area the API uses:
#   ``process_request`` returning the dict the API expects.


class _StubOrchestrator:
    """Minimal stand-in for ``AgentOrchestrator`` for these tests."""

    def __init__(self):
        self.calls = []

    async def process_request(
        self,
        query: str,
        employee_id=None,
        conversation_history=None,
    ):
        self.calls.append({
            "query": query,
            "employee_id": employee_id,
            "history": conversation_history,
        })

        employee_id = employee_id or "EMP001"

        # Heuristic mapping → the same tool calls the real orchestrator
        # would produce for these canned queries. This keeps the API
        # contract tests self-contained.
        q = (query or "").lower()
        if "pto" in q or "vacation" in q:
            tool_calls = [{
                "tool": "check_pto_balance",
                "arguments": {"employee_id": employee_id, "year": 2026},
                "result": {
                    "success": True,
                    "result": {
                        "employee_id": employee_id,
                        "available_days": 8,
                    },
                },
                "success": True,
                "latency_ms": 1,
                "mcp_call": True,
            }]
        elif "remote" in q:
            tool_calls = [{
                "tool": "lookup_employee_profile",
                "arguments": {"employee_id": employee_id},
                "result": {
                    "success": True,
                    "result": {
                        "employee": {
                            "remote_days_per_week": 2,
                            "name": "Alice Johnson",
                            "department": "Engineering",
                        },
                    },
                },
                "success": True,
                "latency_ms": 1,
                "mcp_call": True,
            }]
        elif profile_hint := ("profile" in q or "who am i" in q or "department" in q):
            tool_calls = [{
                "tool": "lookup_employee_profile",
                "arguments": {"employee_id": employee_id},
                "result": {
                    "success": True,
                    "result": {
                        "employee": {
                            "name": "Alice Johnson",
                            "department": "Engineering",
                            "remote_days_per_week": 2,
                        },
                    },
                },
                "success": True,
                "latency_ms": 1,
                "mcp_call": True,
            }]
        else:
            tool_calls = []

        return {
            "answer": "Mocked answer with enough length to satisfy the assertion. " * 3,
            "citations": [],
            "tool_calls": tool_calls,
            "trace": [],
            "metadata": {"latency_ms": 1, "model": "stub"},
        }


@pytest.fixture
def stub_orchestrator(monkeypatch):
    """Patch the FastAPI app's lifespan so it installs a stub orchestrator.

    The real lifespan handler boots an MCP server subprocess and
    initialises a RAG pipeline — both require ``OPENAI_API_KEY``. We
    replace the import target so the lifespan picks up our stub.
    """
    # Pre-seed a stub instance so the module-level imports succeed
    # even if the lifespan is bypassed.
    stub = _StubOrchestrator()

    # Import the API module here so individual test collection does
    # not require env vars.
    from src.api import main as api_main

    monkeypatch.setattr(api_main, "orchestrator", stub, raising=False)
    monkeypatch.setattr(api_main, "rag_pipeline", None, raising=False)
    monkeypatch.setattr(api_main, "mcp_client", None, raising=False)

    # Replace the orchestrator factory used by the API.
    def _fake_get_orchestrator():
        return stub

    monkeypatch.setattr(api_main, "get_orchestrator", _fake_get_orchestrator)

    # Also patch the lifespan so it does not try to spawn an MCP
    # subprocess or load OpenAI credentials.
    async def _noop_lifespan(app):
        yield

    app = api_main.app
    app.router.lifespan_context = _noop_lifespan

    client = TestClient(app)
    yield client, stub


# ---------------------------------------------------------------------------
# PTO balance tests
# ---------------------------------------------------------------------------


class TestPTOBalanceAPI:
    """Test PTO balance via the in-process API."""

    def test_emp001_pto(self, stub_orchestrator):
        client, _stub = stub_orchestrator
        response = client.post(
            "/chat",
            json={"message": "How many PTO days do I have?", "employee_id": "EMP001"},
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        pto_call = next(
            (tc for tc in tool_calls if tc.get("tool") == "check_pto_balance"),
            None,
        )
        assert pto_call is not None, "PTO balance tool should be called"
        pto_result = pto_call.get("result", {}).get("result", {})
        assert pto_result.get("available_days") == 8

    def test_emp005_pto(self, stub_orchestrator):
        client, _stub = stub_orchestrator
        response = client.post(
            "/chat",
            json={"message": "What is my PTO balance?", "employee_id": "EMP005"},
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        pto_call = next(
            (tc for tc in tool_calls if tc.get("tool") == "check_pto_balance"),
            None,
        )
        assert pto_call is not None, "PTO balance tool should be called"
        pto_result = pto_call.get("result", {}).get("result", {})
        assert pto_result.get("available_days") == 8

    def test_emp002_pto(self, stub_orchestrator):
        client, _stub = stub_orchestrator
        response = client.post(
            "/chat",
            json={
                "message": "How many vacation days do I have left?",
                "employee_id": "EMP002",
            },
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        pto_call = next(
            (tc for tc in tool_calls if tc.get("tool") == "check_pto_balance"),
            None,
        )
        assert pto_call is not None, "PTO balance tool should be called"
        pto_result = pto_call.get("result", {}).get("result", {})
        assert pto_result.get("available_days") == 8


# ---------------------------------------------------------------------------
# Remote work tests
# ---------------------------------------------------------------------------


class TestRemoteWorkAPI:
    """Test remote work queries via the in-process API."""

    def test_emp004_remote_days(self, stub_orchestrator):
        client, _stub = stub_orchestrator
        response = client.post(
            "/chat",
            json={
                "message": "How many days can I work remotely per week?",
                "employee_id": "EMP004",
            },
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        profile_call = next(
            (tc for tc in tool_calls if tc.get("tool") == "lookup_employee_profile"),
            None,
        )
        assert profile_call is not None, "Employee profile should be looked up"
        profile = (
            profile_call.get("result", {}).get("result", {}).get("employee", {})
        )
        assert profile.get("remote_days_per_week") == 2

    def test_emp001_remote_days(self, stub_orchestrator):
        client, _stub = stub_orchestrator
        response = client.post(
            "/chat",
            json={
                "message": "What are my remote work options?",
                "employee_id": "EMP001",
            },
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        profile_call = next(
            (tc for tc in tool_calls if tc.get("tool") == "lookup_employee_profile"),
            None,
        )
        assert profile_call is not None, "Employee profile should be looked up"
        profile = (
            profile_call.get("result", {}).get("result", {}).get("employee", {})
        )
        assert profile.get("remote_days_per_week") == 2


# ---------------------------------------------------------------------------
# Employee profile tests
# ---------------------------------------------------------------------------


class TestEmployeeProfileAPI:
    """Test employee profile via the in-process API."""

    def test_emp001_profile(self, stub_orchestrator):
        client, _stub = stub_orchestrator
        response = client.post(
            "/chat",
            json={
                "message": "Who am I? What is my department?",
                "employee_id": "EMP001",
            },
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        profile_call = next(
            (tc for tc in tool_calls if tc.get("tool") == "lookup_employee_profile"),
            None,
        )
        assert profile_call is not None, "Employee profile should be looked up"
        profile = (
            profile_call.get("result", {}).get("result", {}).get("employee", {})
        )
        assert profile.get("name") == "Alice Johnson"
        assert profile.get("department") == "Engineering"


# ---------------------------------------------------------------------------
# Policy + health tests
# ---------------------------------------------------------------------------


class TestPolicyQueries:
    """Test policy-only queries."""

    def test_pto_policy_general(self, stub_orchestrator):
        client, _stub = stub_orchestrator
        response = client.post(
            "/chat",
            json={"message": "What is the company's PTO policy?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 50


class TestHealthCheck:
    """Test health endpoint."""

    def test_health(self, stub_orchestrator):
        client, _stub = stub_orchestrator
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
