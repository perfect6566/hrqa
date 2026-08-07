"""Shared pytest fixtures for the HR Policy Assistant test suite.

This conftest:

* Adds ``src/`` to ``sys.path`` so ``import src.x`` works under any
  working directory (mirrors what the test files used to do inline).
* Provides a reusable ``hr_tools`` fixture that points at ``mock_data``
  so tests do not depend on the real-data fallback path.
* Sets safe test defaults for ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``
  so importing modules that construct an OpenAI client at import time
  does not raise.
* Provides a tiny ``FakeOpenAIClient`` that records calls and returns
  pre-canned chat completions, used by orchestrator/planner tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Resolve project root regardless of where pytest is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Test-time environment: keep modules happy when they auto-build an
# OpenAI client at import / instantiation. The fake client below does
# NOT actually call out, so the value just needs to be non-empty.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:0")


# --------------------------------------------------------------------- #
# HRTools fixture                                                        #
# --------------------------------------------------------------------- #

@pytest.fixture
def hr_tools():
    """Reusable HRTools instance pointing at the bundled ``mock_data``."""
    from src.mcp_server.tools import HRTools

    return HRTools(data_dir=str(_PROJECT_ROOT / "mock_data"))


# --------------------------------------------------------------------- #
# Fake OpenAI client                                                     #
# --------------------------------------------------------------------- #

class _FakeMessage:
    """Mimics the OpenAI SDK message object the orchestrator reads."""

    def __init__(self, content: str = "", tool_calls: Optional[List[Dict[str, Any]]] = None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message


class _FakeUsage:
    def __init__(self, prompt: int = 0, completion: int = 0, total: int = 0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _FakeResponse:
    """Mimics the OpenAI SDK ChatCompletion response."""

    def __init__(self, content: str = "", tool_calls: Optional[List[Dict[str, Any]]] = None,
                 usage: Optional[_FakeUsage] = None):
        self.choices = [_FakeChoice(_FakeMessage(content=content, tool_calls=tool_calls))]
        self.usage = usage


class _FakeCompletions:
    """``client.chat.completions.create`` stub."""

    def __init__(self, scripted: List[_FakeResponse]):
        self._scripted = list(scripted)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self._scripted:
            return _FakeResponse(content="")
        return self._scripted.pop(0)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class FakeOpenAIClient:
    """Drop-in replacement for ``openai.OpenAI``.

    Records every call so tests can assert on prompt contents, and
    returns pre-canned responses in FIFO order. If the script runs
    out, an empty ``_FakeResponse`` is returned (the orchestrator
    treats that as a final-answer call).
    """

    def __init__(self, scripted: Optional[List[_FakeResponse]] = None):
        self._completions = _FakeCompletions(scripted or [])
        self.chat = _FakeChat(self._completions)

    # Convenience accessors used by tests
    @property
    def calls(self) -> List[Dict[str, Any]]:
        return self._completions.calls

    def set_scripted(self, responses: List[_FakeResponse]) -> None:
        self._completions._scripted = list(responses)


@pytest.fixture
def fake_openai():
    """Factory fixture returning a fresh ``FakeOpenAIClient``.

    Usage::

        def test_x(fake_openai):
            client = fake_openai([_FakeResponse(content="hi")])
            ...
    """

    def _make(scripted: Optional[List[_FakeResponse]] = None) -> FakeOpenAIClient:
        return FakeOpenAIClient(scripted)

    return _make


# --------------------------------------------------------------------- #
# Stub MCP client                                                        #
# --------------------------------------------------------------------- #

class FakeMCPClient:
    """Minimal MCP client double for the orchestrator's executor."""

    def __init__(self, tools: Optional[List[Dict[str, Any]]] = None,
                 call_results: Optional[Dict[str, Dict[str, Any]]] = None):
        self._tools = tools or []
        self._call_results = call_results or {}
        self.calls: List[Dict[str, Any]] = []

    async def health_check(self) -> Dict[str, Any]:
        return {"connected": True}

    async def list_tools(self) -> List[Dict[str, Any]]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"name": name, "arguments": arguments})
        if name in self._call_results:
            return self._call_results[name]
        return {"success": False, "error": f"no stub for {name}"}


@pytest.fixture
def fake_mcp():
    """Factory fixture returning a fresh ``FakeMCPClient``."""

    def _make(tools: Optional[List[Dict[str, Any]]] = None,
              call_results: Optional[Dict[str, Dict[str, Any]]] = None) -> FakeMCPClient:
        return FakeMCPClient(tools=tools, call_results=call_results)

    return _make


# --------------------------------------------------------------------- #
# Stub RAG pipeline                                                      #
# --------------------------------------------------------------------- #

class FakeRetriever:
    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None):
        self._chunks = chunks or []
        self.calls: List[Dict[str, Any]] = []

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        self.calls.append({"query": query, "k": k})
        return list(self._chunks[:k])


class FakeRAGPipeline:
    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None):
        self.retriever = FakeRetriever(chunks)


@pytest.fixture
def fake_rag():
    """Factory fixture returning a fresh ``FakeRAGPipeline``."""

    def _make(chunks: Optional[List[Dict[str, Any]]] = None) -> FakeRAGPipeline:
        return FakeRAGPipeline(chunks=chunks)

    return _make
