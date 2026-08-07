"""Task planner: builds the prompt and decides intent hints for the agent loop.

The planner no longer drives tool selection directly - that has moved
to the LLM via native tool calling (``executor.py``). What stays here:

* a system prompt that describes the LLM's role and HR-domain rules;
* an intent hint (``should_use_rag_only``) that the orchestrator uses
  to set ``tool_choice`` cheaply for trivial policy lookups;
* helper to assemble the user prompt from RAG chunks and tool results
  so the orchestrator can stay small.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from openai import OpenAI


class TaskPlanner:
    """Builds prompts and exposes lightweight routing helpers.

    The previous JSON-text planning path was removed because native
    function calling is more reliable: the model can no longer claim
    "I will check PTO" without actually emitting a ``check_pto_balance``
    tool call.
    """

    RAG_ONLY_SYSTEM_PROMPT = """You are an HR Policy Assistant. Answer the user's policy question using ONLY the pre-loaded RAG context below. Cite sources as [Source N]. Do NOT call any tools or emit tool-call instructions."""

    SYSTEM_PROMPT = """You are an HR Policy Assistant that helps employees with company policies and procedures.

Your approach:
1. Understand the user's request
2. Use available tools to gather necessary information via MCP
3. Synthesize a comprehensive, cited response
4. Propose appropriate next steps when relevant

Available MCP tools (call via the provided function-calling interface):
- lookup_employee_profile: Look up employee information
- check_pto_balance: Check PTO balance for an employee
- lookup_benefits_status: Look up benefits enrollment
- search_policy_documents: Search policy documents via RAG
- get_policy_section: Get specific policy section
- check_policy_compliance: Check policy compliance
- create_mock_hr_ticket: Create an HR ticket (mock)
- draft_hr_email: Draft an HR email (mock)

Mandatory tool-use rules (call these tools ONLY when the user is asking
about themselves or their own situation — NOT for general policy questions):
- If the user asks about THEMSELVES (contains "I", "my", "me", "我", "我的", "我自己"),
  you MAY call lookup_employee_profile first to identify them.
- For PTO/vacation/leave questions about the USER, call check_pto_balance.
- For benefits/insurance questions about the USER, call lookup_benefits_status.
- For remote work questions about the USER'S eligibility, call check_policy_compliance.
  For general policy questions like "what is the remote work policy" or
  "what are the rules for WFH" → use search_policy_documents / get_policy_section ONLY.
- search_policy_documents is OPTIONAL for grounding policy claims; prefer it when the answer is not obvious from context.
- Only call create_mock_hr_ticket or draft_hr_email when the user explicitly asks.

Always:
- Cite sources from policy documents using [Source N] notation
- Be specific about policy requirements
- Recommend proper channels for actions
- Distinguish between policy facts and suggestions
- Ask clarifying questions when needed
"""

    PTO_KEYWORDS = (
        "pto", "vacation", "paid time", "time off", "days off",
        "holiday", "holidays", "leave", "annual leave", "time away",
        "年假", "假期", "休假", "调休", "事假", "病假",
    )
    BENEFITS_KEYWORDS = (
        "benefits", "insurance", "medical", "dental", "vision",
        "hsa", "fsa", "401k", "health plan",
        "福利", "医保", "社保", "保险", "退休",
    )
    REMOTE_KEYWORDS = (
        "remote", "wfh", "work from home", "telework", "hybrid",
        "work remotely", "from home", "out of state",
        "远程", "在家办公", "居家办公", "远程办公", "在家工作", "混合办公",
    )
    PROFILE_KEYWORDS = (
        "my profile", "my information", "who am i", "my manager",
        "my department", "my title", "my arrangement",
        "我的档案", "我的资料", "我的部门", "我的经理", "我的入职",
    )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        self.model = model

    def should_use_rag_only(self, query: str, employee_id: Optional[str] = None) -> bool:
        """Heuristic: can this query be answered without any MCP tool call?

        Used by the orchestrator as a ``tool_choice`` hint. False positives
        only cost a tool round-trip; false negatives are caught by the
        tool-guard in the orchestrator.
        """
        q = (query or "").lower()

        # Personal-context signals (any language): if the user is asking
        # *about themselves*, the LLM must be allowed to call identity tools.
        personal_signals = (
            "my ", " i ", "me ", "myself",
            "我", "我的", "我自己",
        )
        personal_tokens = (
            "my", "i", "me", "mine",
        )
        has_personal = (
            any(s in q for s in personal_signals)
            or any(t in q.split() for t in personal_tokens)
        )

        # Compliance / eligibility phrasing always requires lookups.
        compliance_signals = (
            "am i", "can i", "do i qualify", "is it allowed", "allowed to",
            "我能不能", "是否可以", "是否允许", "符合", "符合条件",
        )

        if not has_personal and not any(s in q for s in compliance_signals):
            # Pure policy lookup — let RAG handle it without MCP tools.
            return True

        # Personal-context query with an employee → tools required.
        if has_personal:
            return False
        return False

    def get_required_tools(self, query: str) -> List[str]:
        """Keyword-based tool prediction used by the orchestrator's tool guard.

        This is a *safety net*: if the LLM fails to call a required tool
        during the native-tool-calling loop, the orchestrator invokes
        these tools directly. The prediction is best-effort and biased
        toward false positives (over-calling is safer than missing).
        """
        q = (query or "").lower()
        tools: List[str] = []

        tool_keywords = {
            "lookup_employee_profile": self.PROFILE_KEYWORDS
            + ("my", "i", "me", "employee", "emp"),
            "check_pto_balance": self.PTO_KEYWORDS,
            "lookup_benefits_status": self.BENEFITS_KEYWORDS,
            "search_policy_documents": (
                "policy", "rule", "regulation", "guideline", "what",
                "how", "when", "why", "eligible", "allow", "allowed",
            ),
            "check_policy_compliance": (
                "compliant", "compliance", "eligible", "eligibility",
                "allowed", "can i",
            ),
            "create_mock_hr_ticket": (
                "create ticket", "open ticket", "submit request",
                "file a request", "raise a ticket",
            ),
            "draft_hr_email": (
                "draft email", "send email", "compose email", "email hr",
            ),
        }

        for tool, keywords in tool_keywords.items():
            for kw in keywords:
                if kw in q:
                    tools.append(tool)
                    break
        return tools

    def build_user_prompt(
        self,
        query: str,
        tool_results: List[Dict],
        retrieved_chunks: List[Dict],
        employee_context: Dict,
    ) -> str:
        """Assemble the final user-turn prompt for the synthesizer LLM."""
        parts: List[str] = []

        if tool_results:
            chunks = []
            for tr in tool_results:
                result = tr.get("result") or {}
                if result.get("success"):
                    chunks.append(
                        "MCP Tool: {tool}\n"
                        "Arguments: {args}\n"
                        "Result: {result}".format(
                            tool=tr["tool"],
                            args=json.dumps(
                                tr.get("arguments", {}), ensure_ascii=False
                            ),
                            result=json.dumps(result, ensure_ascii=False, default=str),
                        )
                    )
            if chunks:
                parts.append(
                    "MCP TOOL RESULTS (Authoritative Data):\n" + "\n---\n".join(chunks)
                )

        if employee_context:
            parts.append(
                "Employee Information:\n" + json.dumps(
                    employee_context, indent=2, ensure_ascii=False, default=str
                )
            )

        if retrieved_chunks:
            policy_blocks = []
            for i, c in enumerate(retrieved_chunks, 1):
                policy_blocks.append(
                    f"[Source {i}: {c.get('title', 'Unknown')}]\n"
                    f"Section: {c.get('heading', 'N/A')}\n"
                    f"{c.get('content', '')}"
                )
            parts.append(
                "Policy Documents (from RAG):\n\n" + "\n\n---\n\n".join(policy_blocks)
            )

        context = "\n\n".join(parts)
        prompt = (
            f"Context:\n{context}\n\nQuestion: {query}\n\n"
            "Provide a helpful, accurate response with citations. "
            "Use [Source N] notation to cite policy sources."
        )
        return prompt
