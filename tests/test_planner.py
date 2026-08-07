"""Unit tests for ``src.agent.planner.TaskPlanner``.

These tests exercise the planner's pure / easily-mocked surface:

* keyword-driven ``should_use_rag_only`` heuristic
* keyword-driven ``get_required_tools`` prediction
* ``build_user_prompt`` assembly for the synthesizer call

The planner instantiates an OpenAI client in its constructor; we swap
it out with the ``FakeOpenAIClient`` from ``conftest.py`` so no real
network call is made. None of these tests need that client for the
behaviour under test, but we keep the patch in place to mirror how the
orchestrator wires the planner in.
"""

from __future__ import annotations

import pytest

from src.agent.planner import TaskPlanner

from tests.conftest import FakeOpenAIClient


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #

def _make_planner() -> TaskPlanner:
    """Construct a planner whose OpenAI client never makes a call."""
    planner = TaskPlanner.__new__(TaskPlanner)
    planner.client = FakeOpenAIClient()
    planner.model = "gpt-4o-mini"
    return planner


# --------------------------------------------------------------------- #
# should_use_rag_only                                                    #
# --------------------------------------------------------------------- #

class TestShouldUseRagOnly:
    """The planner's cheap routing heuristic for ``tool_choice``."""

    @pytest.mark.parametrize(
        "query",
        [
            "What is the company's PTO policy?",
            "How does the remote work policy work?",
            "What are the rules for WFH?",
            "年假政策是什么?",
            "公司有什么福利?",
        ],
    )
    def test_pure_policy_questions_return_true(self, query: str) -> None:
        """No personal pronouns / no compliance phrasing → RAG-only is safe."""
        planner = _make_planner()
        assert planner.should_use_rag_only(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "How many PTO days do I have?",
            "Can I work from home?",
            "Am I eligible for remote work?",
            "我的年假还有多少天?",
            "我能不能在家办公?",
            "我是否符合远程办公条件?",
        ],
    )
    def test_personal_or_compliance_questions_return_false(self, query: str) -> None:
        """Personal pronouns / compliance phrasing forces MCP tool use."""
        planner = _make_planner()
        assert planner.should_use_rag_only(query) is False

    def test_empty_query_returns_true(self) -> None:
        """Empty query has no personal signals → safe to treat as RAG-only."""
        planner = _make_planner()
        assert planner.should_use_rag_only("") is True

    def test_employee_id_alone_does_not_force_tools(self) -> None:
        """employee_id alone, no personal/compliance signal, stays RAG-only."""
        planner = _make_planner()
        assert planner.should_use_rag_only("general policy question", employee_id="EMP001") is True


# --------------------------------------------------------------------- #
# get_required_tools                                                     #
# --------------------------------------------------------------------- #

class TestGetRequiredTools:
    """Keyword-based prediction used by the orchestrator's tool guard."""

    def test_pto_query(self) -> None:
        planner = _make_planner()
        tools = planner.get_required_tools("How many vacation days do I have?")
        assert "check_pto_balance" in tools

    def test_benefits_query(self) -> None:
        planner = _make_planner()
        tools = planner.get_required_tools("What medical insurance do I have?")
        assert "lookup_benefits_status" in tools

    def test_remote_work_query(self) -> None:
        planner = _make_planner()
        tools = planner.get_required_tools("Am I allowed to work remotely?")
        assert "lookup_employee_profile" in tools
        assert "check_policy_compliance" in tools

    def test_profile_query(self) -> None:
        planner = _make_planner()
        tools = planner.get_required_tools("Who is my manager?")
        assert "lookup_employee_profile" in tools

    def test_general_policy_query_predicts_search(self) -> None:
        planner = _make_planner()
        tools = planner.get_required_tools("What is the policy on parental leave?")
        assert "search_policy_documents" in tools

    def test_create_ticket_query(self) -> None:
        planner = _make_planner()
        # The keyword list uses literal substrings ("open ticket" not
        # "open a ticket") and PROFILE keywords include "me" / "my", so
        # the guard is biased toward false positives. We assert the
        # targeted tool is present regardless of extra matches.
        tools = planner.get_required_tools("Please open ticket for employee EMP001")
        assert "create_mock_hr_ticket" in tools

    def test_draft_email_query(self) -> None:
        planner = _make_planner()
        # The planner's keyword list uses literal substrings like "draft email"
        # and "email hr"; the guard is biased toward false positives, so we
        # only assert the targeted tool is present.
        tools = planner.get_required_tools("Please draft email to HR")
        assert "draft_hr_email" in tools

    def test_empty_query_returns_empty(self) -> None:
        planner = _make_planner()
        assert planner.get_required_tools("") == []

    def test_unrelated_query_returns_empty_or_minimal(self) -> None:
        """A gibberish query should not pull in random tools."""
        planner = _make_planner()
        tools = planner.get_required_tools("foo bar baz qux")
        # No keywords hit → no tool predictions.
        assert tools == []

    def test_no_duplicate_entries(self) -> None:
        """The planner must dedupe tools even when multiple keywords match."""
        planner = _make_planner()
        tools = planner.get_required_tools("my pto vacation time off leave")
        assert len(tools) == len(set(tools))


# --------------------------------------------------------------------- #
# build_user_prompt                                                      #
# --------------------------------------------------------------------- #

class TestBuildUserPrompt:
    """Assemble the synthesizer prompt from tool results + RAG + employee."""

    def _chunks(self):
        return [
            {
                "document_id": "pto-policy",
                "title": "Paid Time Off (PTO) Policy",
                "heading": "Annual Entitlement",
                "content": "Full-time employees accrue 15 days of PTO per year.",
                "metadata": {"source": "policies/pto-policy.md"},
            },
            {
                "document_id": "remote-work-policy",
                "title": "Remote Work Policy",
                "heading": "Eligibility",
                "content": "Remote eligibility depends on role and tenure.",
                "metadata": {"source": "policies/remote-work-policy.md"},
            },
        ]

    def test_includes_tool_results(self) -> None:
        planner = _make_planner()
        tool_results = [
            {
                "tool": "check_pto_balance",
                "arguments": {"employee_id": "EMP001"},
                "result": {"success": True, "available_days": 8},
            }
        ]
        prompt = planner.build_user_prompt(
            query="How many PTO days do I have?",
            tool_results=tool_results,
            retrieved_chunks=[],
            employee_context={},
        )
        assert "MCP Tool: check_pto_balance" in prompt
        assert "available_days" in prompt
        assert "Question: How many PTO days do I have?" in prompt

    def test_skips_failed_tool_results(self) -> None:
        planner = _make_planner()
        tool_results = [
            {
                "tool": "check_pto_balance",
                "arguments": {"employee_id": "EMP001"},
                "result": {"success": False, "error": "not found"},
            }
        ]
        prompt = planner.build_user_prompt(
            query="q",
            tool_results=tool_results,
            retrieved_chunks=[],
            employee_context={},
        )
        # Failed tool results are filtered out by the planner.
        assert "MCP Tool: check_pto_balance" not in prompt

    def test_includes_employee_context(self) -> None:
        planner = _make_planner()
        prompt = planner.build_user_prompt(
            query="q",
            tool_results=[],
            retrieved_chunks=[],
            employee_context={"name": "Alice", "department": "Engineering"},
        )
        assert "Employee Information:" in prompt
        assert "Alice" in prompt
        assert "Engineering" in prompt

    def test_includes_policy_chunks_with_source_markers(self) -> None:
        planner = _make_planner()
        prompt = planner.build_user_prompt(
            query="q",
            tool_results=[],
            retrieved_chunks=self._chunks(),
            employee_context={},
        )
        assert "Policy Documents (from RAG):" in prompt
        assert "[Source 1:" in prompt
        assert "[Source 2:" in prompt
        assert "Annual Entitlement" in prompt

    def test_handles_missing_optional_inputs(self) -> None:
        planner = _make_planner()
        prompt = planner.build_user_prompt(
            query="q",
            tool_results=[],
            retrieved_chunks=[],
            employee_context={},
        )
        assert "Question: q" in prompt
        assert "Policy Documents" not in prompt
        assert "Employee Information" not in prompt

    def test_tool_results_without_success_key_are_skipped(self) -> None:
        """A tool result missing ``success`` must not break the prompt build."""
        planner = _make_planner()
        tool_results = [
            {
                "tool": "check_policy_compliance",
                "arguments": {},
                "result": {"compliant": True},  # no success key
            }
        ]
        # Should not raise; missing success means we treat it as not-successful.
        prompt = planner.build_user_prompt(
            query="q",
            tool_results=tool_results,
            retrieved_chunks=[],
            employee_context={},
        )
        assert "MCP Tool: check_policy_compliance" not in prompt


# --------------------------------------------------------------------- #
# System prompts                                                         #
# --------------------------------------------------------------------- #

class TestSystemPrompts:
    """Sanity-check the two system prompts stay non-empty and informative."""

    def test_main_system_prompt_mentions_tools(self) -> None:
        assert "MCP" in TaskPlanner.SYSTEM_PROMPT
        # All advertised tools should be referenced.
        for tool in [
            "lookup_employee_profile",
            "check_pto_balance",
            "lookup_benefits_status",
            "search_policy_documents",
        ]:
            assert tool in TaskPlanner.SYSTEM_PROMPT

    def test_rag_only_system_prompt_isolates_rag(self) -> None:
        assert "RAG" in TaskPlanner.RAG_ONLY_SYSTEM_PROMPT
        assert "Do NOT call any tools" in TaskPlanner.RAG_ONLY_SYSTEM_PROMPT
