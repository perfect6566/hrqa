"""Generator for producing RAG-based responses."""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI


class Generator:
    """Generate responses using retrieved context and LLM."""

    SYSTEM_PROMPT = """You are an HR Policy Assistant helping employees with their questions about company policies and procedures.

Your role is to:
1. Answer HR policy questions based on the provided context
2. Provide accurate, helpful information from company documents
3. Cite your sources using [Source N: filename] format — ALWAYS include the filename, e.g., "[Source 1: remote-work-policy.md]", "[Source 2: pto-policy.md]"
4. Distinguish between policy facts and recommendations
5. Escalate complex issues to HR when appropriate

    Guidelines:
- Only answer questions related to company HR policies
- If information is not in the provided context, say so clearly
- Do not make up information or provide advice beyond what's in the policies
- Be professional, helpful, and concise
- For actions requiring approval, recommend the proper process
- CRITICAL: Every time you reference a source, you MUST include the filename after a colon, e.g., "[Source 1: remote-work-policy.md]" not just "[Source 1]"
"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000
    ):
        """
        Initialize the generator.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key
            base_url: OpenAI base URL
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum response length (default 4000 for detailed policy answers)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        # Diagnostic: surface the resolved LLM endpoint so config drift
        # between local .env and the Render dashboard is visible at
        # startup rather than surfacing as a confusing
        # ``UnsupportedProtocol`` deep in the OpenAI SDK on first use.
        effective_key_prefix = (api_key or "")[:7]
        effective_base = base_url or "<unset - defaults to api.openai.com>"
        print(
            f"[generator] LLM config: model={self.model!r} "
            f"base_url={effective_base!r} api_key_prefix={effective_key_prefix!r}"
        )
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; generator cannot reach the LLM."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

    def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response based on query and retrieved context.

        Args:
            query: User's question
            context: Retrieved document chunks
            conversation_history: Optional previous conversation turns

        Returns:
            Dictionary with response, citations, and metadata
        """
        if not context:
            return self._generate_out_of_scope_response(query)

        # Build context string
        context_str = self._build_context(context)

        # Build messages
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {query}"
        })

        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        answer = response.choices[0].message.content

        # Build source mapping for summary
        source_map = {}
        for i, chunk in enumerate(context, 1):
            chunk_meta = chunk.get("metadata", {}) or {}
            filename = chunk_meta.get("filename") or chunk.get("document_id", "unknown")
            source_map[i] = filename

        # Add source summary to answer if sources were referenced
        sources_in_answer = set()
        for i in range(1, len(context) + 1):
            if f"[Source {i}]" in answer:
                sources_in_answer.add(i)

        if sources_in_answer:
            # Add clear source mapping at the end
            summary_lines = ["\n\n---\n**Sources:**"]
            for i in sorted(sources_in_answer):
                summary_lines.append(f"- [Source {i}]: {source_map.get(i, 'unknown')}")
            answer += "\n".join(summary_lines)

        return {
            "answer": answer,
            "citations": self._extract_citations(context, answer),
            "sources_used": [c.get("document_id") for c in context],
            "tokens_used": response.usage.total_tokens if response.usage else 0,
            "model": self.model,
        }

    def generate_with_guardrails(
        self,
        query: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response with explicit guardrails checking.

        Args:
            query: User's question
            context: Retrieved document chunks
            conversation_history: Optional previous conversation turns

        Returns:
            Dictionary with response, citations, safety info
        """
        # Check if query is in-scope
        is_in_scope, reason = self._check_in_scope(query)

        if not is_in_scope:
            return {
                "answer": self._get_out_of_scope_message(reason),
                "citations": [],
                "sources_used": [],
                "in_scope": False,
                "reason": reason,
                "guardrails_applied": True,
            }

        # Generate response
        result = self.generate(query, context, conversation_history)
        result["in_scope"] = True
        result["guardrails_applied"] = True

        # Additional safety checks on response
        result["safety"] = self._check_response_safety(result["answer"])

        return result

    def _build_context(self, context: List[Dict[str, Any]]) -> str:
        """Build formatted context string from chunks."""
        parts = []

        for i, chunk in enumerate(context, 1):
            chunk_meta = chunk.get("metadata", {}) or {}
            # Prefer filename (e.g. "remote-work-policy.md"), fallback to document_id
            filename = chunk_meta.get("filename") or chunk.get("document_id", "unknown")
            title = chunk.get("title", filename)
            source_info = f"[Source {i}: {filename}]"
            if chunk.get("heading"):
                source_info += f" ({chunk['heading']})"
            source_info += f"\n{title}"

            content = chunk.get("content", "").strip()
            parts.append(f"{source_info}\n{content}")

        return "\n\n---\n\n".join(parts)

    def _extract_citations(
        self,
        context: List[Dict[str, Any]],
        answer: str
    ) -> List[Dict[str, Any]]:
        """Extract citations from the generated answer."""
        citations = []

        for i, chunk in enumerate(context, 1):
            # Check if this source was cited
            citation_marker = f"[Source {i}]"
            if citation_marker in answer or f"[{i}]" in answer:
                chunk_meta = chunk.get("metadata", {}) or {}
                citations.append({
                    "source_number": i,
                    "document_id": chunk.get("document_id"),
                    "title": chunk.get("title"),
                    "heading": chunk.get("heading"),
                    "source": chunk_meta.get("source", ""),
                    "filename": chunk_meta.get("filename", ""),
                    "snippet": chunk.get("content", "")[:300],
                    "score": chunk.get("score"),
                })

        return citations

    def _check_in_scope(self, query: str) -> tuple[bool, Optional[str]]:
        """Check if query is within HR policy scope."""
        query_lower = query.lower()

        in_scope_keywords = [
            "pto", "vacation", "holiday", "sick", "leave",
            "remote", "work from home", "hybrid",
            "expense", "reimbursement",
            "benefits", "insurance", "health", "dental", "vision",
            "401k", "retirement", "hsa", "fsa",
            "policy", "procedure", "guideline",
            "onboarding", "orientation",
            "equipment", "laptop", "computer",
            "security", "password", "vpn",
            "harassment", "conduct", "workplace",
            "parental", "maternity", "paternity",
            "manager", "employee", "department",
            "salary", "pay", "compensation",
        ]

        out_of_scope_keywords = [
            "stock", "stocks", "equity", "ipo",
            "merger", "acquisition", "buyout",
            "legal advice", "medical advice",
            "therapy", "mental health treatment",
            "competitor", "confidential business",
        ]

        # Check for out-of-scope first
        for keyword in out_of_scope_keywords:
            if keyword in query_lower:
                return False, f"Questions about {keyword} are outside the scope of HR policy assistance."

        # Check if any in-scope keyword is present
        for keyword in in_scope_keywords:
            if keyword in query_lower:
                return True, None

        return False, "This question doesn't appear to be related to HR policies or procedures."

    def _get_out_of_scope_message(self, reason: Optional[str]) -> str:
        """Get appropriate message for out-of-scope queries."""
        base_message = "I'm here to help with HR-related questions about our company policies and procedures. "

        if reason:
            return base_message + reason

        return base_message + (
            "I can help with topics like PTO, benefits, remote work, expenses, "
            "workplace conduct, and other HR policies. "
            "Is there an HR topic I can assist you with?"
        )

    def _generate_out_of_scope_response(self, query: str) -> Dict[str, Any]:
        """Generate response when no relevant context is found."""
        return {
            "answer": (
                "I couldn't find relevant information in our policy documents to answer your question. "
                "This might be because:\n\n"
                "1. The topic may not be covered in our current policy documents\n"
                "2. The question might be outside the scope of HR policies\n"
                "3. The query might need clarification\n\n"
                "For specific questions, please contact HR directly at hr@company.com "
                "or submit a ticket through the HR portal."
            ),
            "citations": [],
            "sources_used": [],
            "in_scope": None,
            "guardrails_applied": True,
        }

    def _check_response_safety(self, answer: str) -> Dict[str, Any]:
        """Check if response meets safety guidelines."""
        warnings = []

        # Check for potential issues
        if len(answer) > 1500:
            # Long answers should be reviewed
            warnings.append("Long response - verify accuracy")

        return {
            "passed": len(warnings) == 0,
            "warnings": warnings,
        }
