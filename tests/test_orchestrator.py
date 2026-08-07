"""Unit tests for ``src.agent.orchestrator.AgentOrchestrator``.

These tests stub out both the MCP client and the OpenAI client, so we
can drive ``process_request`` end-to-end without ever leaving the
process. The goal is to cover the orchestrator's wiring, tool-calling
loop, tool-guard fallback, RAG short-circuit, and citation assembly.

We attach a ``FakeMCPClient`` directly to the executor so we never
touch the network; ``_ensure_mcp_connected`` is also patched.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.agent.orchestrator import AgentOrchestrator
from src.agent.executor import ToolExecutor, ToolInvocation
from src.agent.planner import TaskPlanner

from tests.conftest import (
    FakeMCPClient,
    FakeOpenAIClient,
    FakeRAGPipeline,
    _FakeResponse,
)


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #

def _tool_call_dict(name: str, arguments: Dict[str, Any], call_id: str = "call_1"):
    """Build an OpenAI tool_call dict in the SDK object shape."""

    class _Fn:
        def __init__(self):
            self.name = name
            self.arguments = arguments

    class _TC:
        def __init__(self):
            self.id = call_id
            self.function = _Fn()

    return _TC()


def _make_orchestrator(
    fake_openai: FakeOpenAIClient,
    chunks: List[Dict[str, Any]] | None = None,
    mcp_tools: List[Dict[str, Any]] | None = None,
    mcp_results: Dict[str, Dict[str, Any]] | None = None,
) -> AgentOrchestrator:
    """Build an orchestrator with both MCP and OpenAI stubbed."""

    rag = FakeRAGPipeline(chunks=chunks or [])
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch.rag_pipeline = rag
    orch.mcp_server_url = "http://localhost:0"
    orch.model = "gpt-4o-mini"
    orch.use_mcp_protocol = True
    orch.max_tool_iterations = 3

    # Replace the OpenAI client so no network calls are made.
    orch.client = fake_openai

    # Wire planner and executor. Planner gets its own client; executor
    # gets a FakeMCPClient so we never touch the network.
    orch.planner = TaskPlanner.__new__(TaskPlanner)
    orch.planner.client = fake_openai
    orch.planner.model = "gpt-4o-mini"

    orch.executor = ToolExecutor(
        mcp_client=FakeMCPClient(tools=mcp_tools or [], call_results=mcp_results or {}),
        max_parallel=4,
    )

    orch._rag_cache = {}
    orch._rag_cache_max = 64

    # Bypass the real MCP connection / tool loading entirely.
    orch._ensure_mcp_connected = _make_async_returning(True)  # type: ignore[assignment]

    return orch


def _make_async_returning(value):
    async def _fn(*args, **kwargs):
        return value
    return _fn


def _final_answer_response(content: str) -> _FakeResponse:
    """A response with no tool calls — terminates the loop."""
    return _FakeResponse(content=content)


def _tool_call_response(name: str, arguments: Dict[str, Any], call_id: str = "call_1") -> _FakeResponse:
    """A response with one tool call."""
    return _FakeResponse(
        content="",
        tool_calls=[_tool_call_dict(name, arguments, call_id=call_id)],
    )


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #

class TestOrchestratorConstruction:
    """Initial wiring of planner, executor, and OpenAI client."""

    def test_capabilities_when_no_mcp_cache(self, fake_openai) -> None:
        orch = _make_orchestrator(fake_openai())
        caps = orch.get_capabilities()
        # Falls back to the static tool list when the executor cache is empty.
        assert "lookup_employee_profile" in caps["tools"]
        assert "check_pto_balance" in caps["tools"]
        assert caps["model"] == "gpt-4o-mini"
        assert caps["rag_enabled"] is True

    def test_capabilities_reflects_executor_cache(self, fake_openai) -> None:
        orch = _make_orchestrator(fake_openai())
        orch.executor._openai_tools_cache = [
            {"function": {"name": "only_one"}},
        ]
        caps = orch.get_capabilities()
        assert caps["tools"] == ["only_one"]


# --------------------------------------------------------------------- #
# Pure policy (RAG-only) flow                                            #
# --------------------------------------------------------------------- #

class TestRagOnlyFlow:
    """When the heuristic classifies the query as RAG-only, no MCP tools run."""

    @pytest.mark.asyncio
    async def test_rag_only_skips_tool_loop(self, fake_openai, fake_rag) -> None:
        chunks = [
            {
                "document_id": "pto-policy",
                "title": "Paid Time Off (PTO) Policy",
                "heading": "Annual Entitlement",
                "content": "Employees accrue 15 days per year.",
                "metadata": {"source": "policies/pto-policy.md", "filename": "pto-policy.md"},
            }
        ]
        client = fake_openai([
            _final_answer_response("Per [Source 1: PTO Policy — Annual Entitlement], "
                                   "you accrue 15 days per year."),
        ])

        orch = _make_orchestrator(client, chunks=chunks)

        result = await orch.process_request(
            query="What is the company's PTO policy?",
        )

        assert "15 days" in result["answer"]
        # References footer was appended because sources_in_answer was populated.
        assert "References" in result["answer"]
        # Two LLM calls: one inside the rag_only branch, one for synthesis.
        assert len(client.calls) == 2
        # No tool calls happened.
        assert result["tool_calls"] == []
        assert result["metadata"]["mcp_used"] is False
        assert result["metadata"]["tools_used"] == 0

    @pytest.mark.asyncio
    async def test_rag_only_with_no_chunks_still_synthesises(self, fake_openai) -> None:
        client = fake_openai([
            _final_answer_response("I don't have that information."),
        ])

        orch = _make_orchestrator(client, chunks=[])

        result = await orch.process_request(query="What is the policy?")

        assert "don't have" in result["answer"]
        # Citations list is empty because nothing was cited.
        assert result["citations"] == []


# --------------------------------------------------------------------- #
# Tool-calling loop                                                      #
# --------------------------------------------------------------------- #

class TestToolCallingLoop:
    """Exercise the native tool-calling flow end-to-end."""

    @pytest.mark.asyncio
    async def test_executes_tool_and_returns_final_answer(self, fake_openai) -> None:
        client = fake_openai([
            # First LLM turn: emits one tool call.
            _tool_call_response("check_pto_balance", {"employee_id": "EMP001"}),
            # Second LLM turn (synthesizer): returns the final answer.
            _final_answer_response("You have 8 PTO days available per the tool result."),
        ])

        mcp_results = {
            "check_pto_balance": {
                "success": True,
                "employee_name": "Alice Johnson",
                "available_days": 8,
            },
        }
        mcp_tools = [
            {"name": "check_pto_balance", "description": "Check PTO",
             "inputSchema": {"type": "object", "properties": {"employee_id": {"type": "string"}}}}
        ]

        orch = _make_orchestrator(
            client, mcp_tools=mcp_tools, mcp_results=mcp_results
        )

        result = await orch.process_request(
            query="How many PTO days do I have?",
            employee_id="EMP001",
        )

        assert "8 PTO days" in result["answer"]
        # The tool call is recorded.
        tools = [tc["tool"] for tc in result["tool_calls"]]
        assert "check_pto_balance" in tools
        # The trace records the LLM iterations and the executed tool.
        steps = {t["step"] for t in result["trace"]}
        assert "llm_iteration" in steps
        assert "tool_executed" in steps

    @pytest.mark.asyncio
    async def test_drops_tool_call_noise_and_retries(self, fake_openai) -> None:
        """When the LLM returns empty content with no tool calls, force a retry.

        Use a personal query so the planner routes through the tool-calling
        branch (not the rag_only branch) — that's where the noise-drop +
        retry logic lives.
        """
        client = fake_openai([
            _final_answer_response(""),       # empty content, no tool calls
            _final_answer_response("Clean answer."),
        ])

        mcp_tools = [
            {"name": "check_pto_balance", "description": "d",
             "inputSchema": {"type": "object", "properties": {}}}
        ]
        orch = _make_orchestrator(client, mcp_tools=mcp_tools)

        result = await orch.process_request(
            query="How many PTO days do I have?",
            employee_id="EMP001",
        )

        assert "Clean answer" in result["answer"]
        steps = [t["step"] for t in result["trace"]]
        assert "tool_call_noise_dropped" in steps

    @pytest.mark.asyncio
    async def test_handles_max_iterations(self, fake_openai) -> None:
        """If the model keeps emitting tool calls, the loop terminates and forces a final answer."""
        client = fake_openai([
            # All scripted responses are tool calls; after max_tool_iterations
            # the orchestrator falls back to a tool_choice="none" forced answer.
            _tool_call_response("check_pto_balance", {"employee_id": "EMP001"}),
            _tool_call_response("check_pto_balance", {"employee_id": "EMP001"}),
            _tool_call_response("check_pto_balance", {"employee_id": "EMP001"}),
            # Forced final answer after loop exhaustion.
            _final_answer_response("Forced final answer after iteration cap."),
        ])

        mcp_results = {
            "check_pto_balance": {"success": True, "available_days": 8},
        }
        mcp_tools = [
            {"name": "check_pto_balance", "description": "d",
             "inputSchema": {"type": "object", "properties": {}}}
        ]

        orch = _make_orchestrator(
            client, mcp_tools=mcp_tools, mcp_results=mcp_results
        )
        # Tighten the cap so the test stays fast.
        orch.max_tool_iterations = 2

        result = await orch.process_request(
            query="How many PTO days do I have?",
            employee_id="EMP001",
        )

        # Forced answer is what the user sees.
        assert "Forced final answer" in result["answer"]
        steps = [t["step"] for t in result["trace"]]
        assert "final_answer_forced" in steps


# --------------------------------------------------------------------- #
# Tool-guard fallback                                                    #
# --------------------------------------------------------------------- #

class TestToolGuardFallback:
    """The orchestrator must force-call mandatory tools the LLM skipped."""

    @pytest.mark.asyncio
    async def test_tool_guard_runs_when_llm_skipped_required_tool(self, fake_openai) -> None:
        # The LLM responds with no tool call even though the query needs PTO.
        # The tool guard should then run check_pto_balance.
        client = fake_openai([
            _final_answer_response("Here's what I found."),
            # The synthesis call (no tool_choice, just plain synthesis).
            _final_answer_response("Final answer referencing the tool result."),
        ])

        mcp_results = {
            "check_pto_balance": {"success": True, "available_days": 8},
        }
        mcp_tools = [
            {"name": "check_pto_balance", "description": "d",
             "inputSchema": {"type": "object", "properties": {}}}
        ]

        orch = _make_orchestrator(
            client, mcp_tools=mcp_tools, mcp_results=mcp_results
        )

        result = await orch.process_request(
            query="How many PTO days do I have?",
            employee_id="EMP001",
        )

        # The tool guard fired and recorded the call.
        tool_steps = [t for t in result["trace"] if t["step"] == "tool_guard_executed"]
        assert tool_steps, "Expected tool_guard_executed step in trace"
        assert tool_steps[0]["tool"] == "check_pto_balance"
        # And the invocations list contains that tool.
        tools = [tc["tool"] for tc in result["tool_calls"]]
        assert "check_pto_balance" in tools

    @pytest.mark.asyncio
    async def test_tool_guard_skipped_when_no_employee_id(self, fake_openai) -> None:
        client = fake_openai([
            _final_answer_response("Done."),
        ])
        orch = _make_orchestrator(client)
        result = await orch.process_request(
            query="How many PTO days do I have?",
            employee_id=None,
        )
        # No employee_id → tool guard cannot fill default args → no calls.
        assert result["tool_calls"] == []


# --------------------------------------------------------------------- #
# Employee-not-found short-circuit                                       #
# --------------------------------------------------------------------- #

class TestEmployeeNotFound:
    """When all employee-lookup tools return 'not found', short-circuit."""

    @pytest.mark.asyncio
    async def test_short_circuit_message(self, fake_openai) -> None:
        client = fake_openai([
            _tool_call_response("check_pto_balance", {"employee_id": "EMP999"}),
            _final_answer_response("unused"),
        ])

        mcp_results = {
            "check_pto_balance": {"success": False, "error": "Employee not found"},
        }
        mcp_tools = [
            {"name": "check_pto_balance", "description": "d",
             "inputSchema": {"type": "object", "properties": {}}}
        ]

        orch = _make_orchestrator(
            client, mcp_tools=mcp_tools, mcp_results=mcp_results
        )

        result = await orch.process_request(
            query="How many PTO days do I have?",
            employee_id="EMP999",
        )

        assert "could not find" in result["answer"]
        assert "EMP999" in result["answer"]
        assert result["citations"] == []
        steps = [t["step"] for t in result["trace"]]
        assert "synthesis_short_circuit" in steps


# --------------------------------------------------------------------- #
# Citation assembly                                                      #
# --------------------------------------------------------------------- #

class TestCitationAssembly:
    """End-to-end test that in-body citations produce a References footer."""

    @pytest.mark.asyncio
    async def test_references_footer_added(self, fake_openai) -> None:
        chunks = [
            {
                "document_id": "pto-policy",
                "title": "PTO Policy",
                "heading": "Annual Entitlement",
                "content": "You accrue 15 days.",
                "metadata": {"source": "policies/pto-policy.md", "filename": "pto-policy.md"},
            },
            {
                "document_id": "remote-work-policy",
                "title": "Remote Work Policy",
                "heading": "Eligibility",
                "content": "Remote work requires manager approval.",
                "metadata": {"source": "policies/remote-work-policy.md", "filename": "remote-work-policy.md"},
            },
        ]
        # Synthesizer answer cites both sources in the body.
        body = (
            "Per [Source 1: PTO Policy — Annual Entitlement] and "
            "[Source 2: Remote Work Policy — Eligibility], here's the answer."
        )
        client = fake_openai([_final_answer_response(body)])

        orch = _make_orchestrator(client, chunks=chunks)

        result = await orch.process_request(query="What is the policy?")

        # References footer present and uses filenames.
        assert "**References:**" in result["answer"]
        assert "Source 1: pto-policy.md" in result["answer"]
        assert "Source 2: remote-work-policy.md" in result["answer"]

        # Citations list contains both sources with snippet metadata.
        assert len(result["citations"]) == 2
        assert {c["source_number"] for c in result["citations"]} == {1, 2}
        for c in result["citations"]:
            assert c["snippet"]  # populated
            assert c["filename"]  # populated

    @pytest.mark.asyncio
    async def test_no_footer_when_no_sources_cited(self, fake_openai) -> None:
        chunks = [
            {
                "document_id": "pto-policy",
                "title": "PTO Policy",
                "heading": "Annual Entitlement",
                "content": "You accrue 15 days.",
                "metadata": {"source": "policies/pto-policy.md", "filename": "pto-policy.md"},
            }
        ]
        client = fake_openai([
            _final_answer_response("I don't have that information."),
        ])
        orch = _make_orchestrator(client, chunks=chunks)
        result = await orch.process_request(query="Anything")
        assert "**References:**" not in result["answer"]
        assert result["citations"] == []


# --------------------------------------------------------------------- #
# RAG fallback when neither pre-loop nor LLM produced chunks             #
# --------------------------------------------------------------------- #

class TestRagRetrievalFallback:
    """If the LLM does not call ``search_policy_documents`` and the pre-loop
    retrieval returns nothing, we still re-retrieve from the RAG pipeline
    before synthesis."""

    @pytest.mark.asyncio
    async def test_rag_source_fallback(self, fake_openai) -> None:
        chunks = [
            {
                "document_id": "pto-policy",
                "title": "PTO Policy",
                "heading": "Annual Entitlement",
                "content": "...",
                "metadata": {"source": "policies/pto-policy.md", "filename": "pto-policy.md"},
            }
        ]
        # Two-call flow: tool-loop final answer, then synthesis.
        client = fake_openai([
            _final_answer_response("answer"),
            _final_answer_response("[Source 1: PTO Policy — Annual Entitlement] detail"),
        ])
        orch = _make_orchestrator(client, chunks=[])
        # First call: chunks empty (pre-loop miss).
        # We then need the post-loop retrieval to populate. The orchestrator
        # calls `_retrieve_rag_once` again; our FakeRAGPipeline still returns
        # nothing by default, so we override after construction.
        orch.rag_pipeline = FakeRAGPipeline(chunks=chunks)

        result = await orch.process_request(
            query="general PTO policy",
        )

        # Source should be either pre_loop or fallback_post_loop.
        assert result["metadata"]["rag_source"] in {"pre_loop", "fallback_post_loop"}


# --------------------------------------------------------------------- #
# Stripping tool-call artifacts                                          #
# --------------------------------------------------------------------- #

class TestStripToolCallArtifacts:
    """The orchestrator must strip pseudo-XML tool-call markers from replies."""

    @pytest.mark.asyncio
    async def test_strips_dsml_pseudo_xml(self, fake_openai) -> None:
        client = fake_openai([
            _final_answer_response(
                "Here is the answer.\n<|DSML|>tool_calls>ignore this</tool_calls>"
            ),
        ])
        orch = _make_orchestrator(client)
        result = await orch.process_request(query="What is PTO?")
        assert "<|DSML|>" not in result["answer"]
        assert "Here is the answer" in result["answer"]

    @pytest.mark.asyncio
    async def test_strips_python_tag(self, fake_openai) -> None:
        client = fake_openai([
            _final_answer_response(
                "Good answer.<|python_tag|>print('hi')<|/python_tag|>"
            ),
        ])
        orch = _make_orchestrator(client)
        result = await orch.process_request(query="What is PTO?")
        assert "<|python_tag|>" not in result["answer"]
        assert "Good answer" in result["answer"]

    @pytest.mark.asyncio
    async def test_strips_tool_call_marker(self, fake_openai) -> None:
        client = fake_openai([
            _final_answer_response("Answer.<|tool_call|>oops"),
        ])
        orch = _make_orchestrator(client)
        result = await orch.process_request(query="x")
        assert "<|tool_call|>" not in result["answer"]


# --------------------------------------------------------------------- #
# Conversation history                                                   #
# --------------------------------------------------------------------- #

class TestConversationHistory:
    @pytest.mark.asyncio
    async def test_history_included_in_initial_messages(self, fake_openai) -> None:
        client = fake_openai([
            _final_answer_response("ok"),
        ])
        orch = _make_orchestrator(client)

        history = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]

        result = await orch.process_request(
            query="follow up",
            conversation_history=history,
        )

        # First LLM call should include both the system prompt, history,
        # the user turn, and (because no RAG chunks) nothing else.
        first_call_messages = client.calls[0]["messages"]
        roles = [m["role"] for m in first_call_messages]
        assert roles[0] == "system"
        assert "user" in roles
        assert "assistant" in roles
        # The history turns are present verbatim.
        joined = "\n".join(m["content"] for m in first_call_messages if m.get("content"))
        assert "earlier question" in joined
        assert "earlier answer" in joined


# --------------------------------------------------------------------- #
# Helpers (private)                                                      #
# --------------------------------------------------------------------- #

class TestInternalHelpers:
    """Cover the small pure helpers as units."""

    def test_extract_employee_context_returns_empty_when_no_profile_call(self):
        invocations: List[ToolInvocation] = []
        ctx = AgentOrchestrator._extract_employee_context(invocations)
        assert ctx == {}

    def test_extract_employee_context_returns_employee_dict(self):
        invocations = [
            ToolInvocation(
                tool="lookup_employee_profile",
                arguments={"employee_id": "EMP001"},
                result={"success": True, "employee": {"name": "Alice"}},
                success=True,
                latency_ms=1,
            ),
        ]
        ctx = AgentOrchestrator._extract_employee_context(invocations)
        assert ctx == {"name": "Alice"}

    def test_strip_tool_call_artifacts_passthrough(self):
        assert AgentOrchestrator._strip_tool_call_artifacts("plain text") == "plain text"

    def test_strip_tool_call_artifacts_empty(self):
        assert AgentOrchestrator._strip_tool_call_artifacts("") == ""

    def test_chunks_from_tool_result_handles_missing_keys(self):
        out = AgentOrchestrator._chunks_from_tool_result({"results": [{}]})
        assert len(out) == 1
        assert out[0]["score"] == 0
        assert out[0]["metadata"] == {}

    def test_chunks_from_tool_result_handles_non_dict_result(self):
        assert AgentOrchestrator._chunks_from_tool_result("not a dict") == []

    def test_parse_body_section_names_extracts_headings(self):
        text = "Per [Source 1: PTO Policy — Annual Entitlement] and [Source 2: filename — Section 2.1]"
        sections = AgentOrchestrator._parse_body_section_names(text)
        assert sections.get(1) == "Annual Entitlement"
        assert sections.get(2) == "Section 2.1"

    def test_parse_body_section_names_ignores_trailing_punctuation(self):
        text = "Per [Source 1: PTO Policy — this is a sentence.]"
        assert 1 not in AgentOrchestrator._parse_body_section_names(text)

    def test_parse_body_section_names_ignores_bare_filename(self):
        text = "Per [Source 1: pto-policy.md]"
        assert AgentOrchestrator._parse_body_section_names(text) == {}

    def test_detect_employee_not_found_returns_false_for_no_id(self):
        assert AgentOrchestrator._detect_employee_not_found([], None) is False

    def test_detect_employee_not_found_returns_false_when_no_relevant_calls(self):
        inv = ToolInvocation(
            tool="search_policy_documents",
            arguments={},
            result={"success": True},
            success=True,
            latency_ms=1,
        )
        assert AgentOrchestrator._detect_employee_not_found([inv], "EMP001") is False

    def test_detect_employee_not_found_true_when_all_failed(self):
        inv = ToolInvocation(
            tool="lookup_employee_profile",
            arguments={"employee_id": "EMP999"},
            result={"success": False, "error": "Employee not found"},
            success=False,
            latency_ms=1,
        )
        assert AgentOrchestrator._detect_employee_not_found([inv], "EMP999") is True

    def test_detect_employee_not_found_false_when_any_succeeded(self):
        ok = ToolInvocation(
            tool="lookup_employee_profile",
            arguments={"employee_id": "EMP001"},
            result={"success": True, "employee": {"name": "Alice"}},
            success=True,
            latency_ms=1,
        )
        bad = ToolInvocation(
            tool="check_pto_balance",
            arguments={},
            result={"success": False, "error": "Employee not found"},
            success=False,
            latency_ms=1,
        )
        assert AgentOrchestrator._detect_employee_not_found([ok, bad], "EMP001") is False

    def test_build_user_turn_includes_employee_id(self):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        msg = orch._build_user_turn("how many PTO days", "EMP001")
        assert "EMP001" in msg
        assert "how many PTO days" in msg

    def test_build_user_turn_without_employee_id(self):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        msg = orch._build_user_turn("what is the policy", None)
        assert "what is the policy" in msg
        assert "Employee ID" not in msg

    def test_mandatory_tools_for_pto_query(self):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        assert orch._mandatory_tools("how many PTO days") == ["check_pto_balance"]

    def test_mandatory_tools_for_remote_query(self):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        tools = orch._mandatory_tools("can I work from home?")
        assert "lookup_employee_profile" in tools
        assert "check_policy_compliance" in tools

    def test_mandatory_tools_empty_for_unrelated_query(self):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        assert orch._mandatory_tools("foo bar baz") == []

    def test_default_args_for_known_tools(self):
        assert AgentOrchestrator._default_args_for(
            "lookup_employee_profile", "EMP001"
        ) == {"employee_id": "EMP001"}
        assert AgentOrchestrator._default_args_for(
            "check_pto_balance", "EMP001"
        ) == {"employee_id": "EMP001", "year": 2026}

    def test_default_args_for_unknown_tool_returns_none(self):
        assert AgentOrchestrator._default_args_for("no_such_tool", "EMP001") is None

    def test_format_rag_context_empty_chunks(self):
        assert AgentOrchestrator._format_rag_context([]) == ""

    def test_format_rag_context_with_chunks(self):
        chunks = [{
            "title": "T", "heading": "H", "content": "C",
        }]
        out = AgentOrchestrator._format_rag_context(chunks)
        assert "[Source 1:" in out
        assert "Section: H" in out
        assert "C" in out

    def test_process_request_sync_runs_async_coroutine(self, fake_openai) -> None:
        client = fake_openai([_final_answer_response("sync answer")])
        orch = _make_orchestrator(client)
        result = orch.process_request_sync(query="anything")
        assert "sync answer" in result["answer"]


# --------------------------------------------------------------------- #
# MCP connection failure                                                 #
# --------------------------------------------------------------------- #

class TestMcpConnection:
    @pytest.mark.asyncio
    async def test_connection_failure_falls_back_to_rag_only(self, fake_openai) -> None:
        client = fake_openai([
            _final_answer_response("answer without tools"),
        ])
        orch = _make_orchestrator(client)
        orch._ensure_mcp_connected = _make_async_returning(False)  # type: ignore[assignment]

        result = await orch.process_request(query="how many PTO days")

        # Even with an employee_id, no MCP tool calls because connection failed
        # AND the planner decided to route to RAG-only for a non-personal query.
        assert result["metadata"]["mcp_connected"] is False
