"""Agent orchestrator: combines RAG, MCP tool calling, and LLM synthesis.

The orchestrator's job is to drive the tool-calling loop. It no longer
maintains an in-house planner over free-form JSON - the LLM picks
tools directly via OpenAI's native function calling and we execute
them via MCP. The ``executor.ToolExecutor`` owns the low-level
plumbing; ``planner.TaskPlanner`` provides prompt construction and a
small keyword-based tool guard as a safety net.

High-level flow per request:

1. Ensure MCP is connected and load tool schema.
2. Decide ``tool_choice`` from a cheap RAG-only heuristic.
3. Loop: ask LLM with native tool calling -> if it returns
   ``tool_calls``, execute them in parallel via MCP and feed results
   back as ``role: tool`` messages; else return final answer.
4. Enforce the tool guard: any mandatory tool the model skipped
   (PTO/benefits/remote/profile) is force-invoked before synthesising
   the final answer.
5. Run a single RAG retrieval so the LLM has policy chunks even if
   it did not call ``search_policy_documents`` itself, and synthesise
   the final answer with citations.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .executor import ToolExecutor, ToolInvocation
from .planner import TaskPlanner


class AgentOrchestrator:
    """Coordinates RAG + MCP tools + LLM synthesis."""

    def __init__(
        self,
        rag_pipeline,
        mcp_server_url: str = "http://localhost:8001",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_mcp_protocol: bool = True,
        max_tool_iterations: int = 4,
    ):
        self.rag_pipeline = rag_pipeline
        self.mcp_server_url = mcp_server_url
        self.model = model
        self.use_mcp_protocol = use_mcp_protocol
        self.max_tool_iterations = max_tool_iterations

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.planner = TaskPlanner(model=model, api_key=api_key, base_url=base_url)
        self.executor = ToolExecutor(mcp_client=None)  # set after connect

        # Per-instance RAG cache: key=(query, k) -> list of chunks.
        # Sized small so the agent stays a long-lived object without unbounded growth.
        self._rag_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        self._rag_cache_max = 64

    def _retrieve_rag_once(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Single-shot RAG retrieval with a small per-instance cache.

        Replaces the previous "always retrieve after the tool loop" pattern.
        The orchestrator now retrieves once before the tool loop, primes the
        LLM with those chunks, and only re-retrieves if the LLM explicitly
        calls ``search_policy_documents`` with a different query.
        """
        if not self.rag_pipeline or not getattr(self.rag_pipeline, "retriever", None):
            return []
        key = (query or "", k)
        if key in self._rag_cache:
            return self._rag_cache[key]
        try:
            chunks = self.rag_pipeline.retriever.retrieve(query, k=k)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"RAG retrieval failed: {exc}")
            chunks = []
        # Bound cache size to keep long-running agents bounded.
        if len(self._rag_cache) >= self._rag_cache_max:
            self._rag_cache.pop(next(iter(self._rag_cache)))
        self._rag_cache[key] = chunks
        return chunks

    @staticmethod
    def _format_rag_context(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return ""
        blocks = []
        for i, c in enumerate(chunks, 1):
            blocks.append(
                f"[Source {i}: {c.get('title', 'Unknown')}]\n"
                f"Section: {c.get('heading', 'N/A')}\n"
                f"{c.get('content', '')}"
            )
        return "Policy Documents (from RAG):\n\n" + "\n\n---\n\n".join(blocks)

    # ------------------------------------------------------------------ #
    # MCP lifecycle                                                       #
    # ------------------------------------------------------------------ #

    async def _ensure_mcp_connected(self) -> bool:
        """Connect to MCP server, refresh tool cache, and bind the executor."""
        try:
            from ..mcp import fastmcp_client as mcp_client_module
        except ImportError:
            return False

        if self.executor.mcp_client is None:
            self.executor.mcp_client = mcp_client_module.MCPClient(
                server_url=self.mcp_server_url
            )
        try:
            health = await self.executor.mcp_client.health_check()
            if not health.get("connected"):
                return False
            await self.executor.load_openai_tools(refresh=True)
            return True
        except Exception as e:
            print(f"MCP connection failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Main entry point                                                    #
    # ------------------------------------------------------------------ #

    async def process_request(
        self,
        query: str,
        employee_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Process a user request end-to-end via the tool-calling loop."""
        start_time = time.time()
        trace: List[Dict[str, Any]] = []
        invocations: List[ToolInvocation] = []

        trace.append({
            "step": "start",
            "timestamp": datetime.now().isoformat(),
            "input": query,
            "employee_id": employee_id,
        })

        mcp_connected = await self._ensure_mcp_connected()
        trace.append({
            "step": "mcp_connection",
            "timestamp": datetime.now().isoformat(),
            "connected": mcp_connected,
            "url": self.mcp_server_url,
        })

        # ----- build initial messages ---------------------------------
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": TaskPlanner.SYSTEM_PROMPT}
        ]
        if conversation_history:
            messages.extend(conversation_history)

        user_turn = self._build_user_turn(query, employee_id)
        messages.append({"role": "user", "content": user_turn})

        trace.append({
            "step": "messages_initialized",
            "timestamp": datetime.now().isoformat(),
            "history_messages": len(conversation_history or []),
        })

        # ----- tool_choice hint --------------------------------------
        # If the heuristic says this is a pure policy question with no
        # personal context, force the model to answer directly.
        rag_only = self.planner.should_use_rag_only(query, employee_id)
        tool_choice = "none" if rag_only else "auto"

        # ----- pre-loop RAG retrieval (cached) ----------------------
        # Single retrieval that primes the LLM with policy context. If the
        # model later decides to call ``search_policy_documents`` itself,
        # the loop replaces these chunks with the LLM-chosen query's results.
        retrieved_chunks: List[Dict[str, Any]] = self._retrieve_rag_once(query, k=5)
        rag_source = "pre_loop"
        trace.append({
            "step": "rag_pre_loop",
            "timestamp": datetime.now().isoformat(),
            "chunks_retrieved": len(retrieved_chunks),
            "scores": [c.get("score", 0) for c in retrieved_chunks],
        })
        if retrieved_chunks:
            messages.append({
                "role": "assistant",
                "content": (
                    "I have pre-loaded the following policy context for this query. "
                    "You may cite these directly via [Source N], or call "
                    "search_policy_documents if you need a different angle.\n\n"
                    + self._format_rag_context(retrieved_chunks)
                ),
            })

        # ----- native tool-calling loop ------------------------------
        openai_tools = (
            await self.executor.load_openai_tools() if mcp_connected else []
        )

        # When rag_only=True, DeepSeek Flash may still emit tool_calls despite
        # tool_choice="none". Skip any tool execution in that case and let the
        # synthesizer answer from the pre-loaded RAG context.
        if rag_only:
            invocations = []  # no tools needed for pure policy lookups
            messages.append({
                "role": "system",
                "content": (
                    "Answer the user's policy question directly using the pre-loaded "
                    "RAG context below. Cite sources as [Source N]. "
                    "Do NOT call any tools or emit tool-call instructions."
                ),
            })
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=4096,
            )
            final_answer = self._strip_tool_call_artifacts(
                response.choices[0].message.content or ""
            )
            trace.append({
                "step": "llm_iteration",
                "iteration": 1,
                "timestamp": datetime.now().isoformat(),
                "tool_choice": "rag_only_skipped",
                "had_tool_calls": False,
                "latency_ms": int((time.time() - start_time) * 1000),
            })
            retrieved_chunks = self._retrieve_rag_once(query, k=5)
            rag_source = "pre_loop"
            loop_steps = 0
        else:
            final_answer: Optional[str] = None
            loop_steps = 0
            invocations: List[ToolInvocation] = []
            for iteration in range(self.max_tool_iterations):
                loop_steps = iteration + 1
                iter_started = time.time()

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice=tool_choice if openai_tools else None,
                    temperature=0.3,
                    max_tokens=4096,
                )
                msg = response.choices[0].message
                tool_calls = ToolExecutor.parse_tool_calls(msg)

                iter_trace: Dict[str, Any] = {
                    "step": "llm_iteration",
                    "iteration": iteration + 1,
                    "timestamp": datetime.now().isoformat(),
                    "tool_choice": tool_choice,
                    "had_tool_calls": bool(tool_calls),
                    "latency_ms": int((time.time() - iter_started) * 1000),
                }
                if hasattr(response, "usage") and response.usage:
                    iter_trace["tokens"] = {
                        "prompt": getattr(response.usage, "prompt_tokens", None),
                        "completion": getattr(response.usage, "completion_tokens", None),
                        "total": getattr(response.usage, "total_tokens", None),
                    }
                trace.append(iter_trace)

                if not tool_calls:
                    final_answer = self._strip_tool_call_artifacts(msg.content or "")
                    if final_answer:
                        break
                    trace.append({
                        "step": "tool_call_noise_dropped",
                        "timestamp": datetime.now().isoformat(),
                        "iteration": iteration + 1,
                    })
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            *messages,
                            {
                                "role": "system",
                                "content": (
                                    "Reply in plain text only. Do NOT emit any "
                                    "tool-call XML or pseudo-XML."
                                ),
                            },
                        ],
                        tool_choice="none",
                        temperature=0.3,
                        max_tokens=4096,
                    )
                    final_answer = self._strip_tool_call_artifacts(
                        response.choices[0].message.content or ""
                    )
                    break

                # ----- execute tool calls in parallel --------------------
                messages.append(ToolExecutor.assistant_tool_calls_message(msg))

                call_payload = [
                    {"name": tc["name"], "arguments": tc["arguments"]}
                    for tc in tool_calls
                ]
                results = await self.executor.call_many(call_payload)
                invocations.extend(results)

                for tc, inv in zip(tool_calls, results):
                    trace.append({
                        "step": "tool_executed",
                        "timestamp": inv.started_at,
                        "tool": inv.tool,
                        "arguments": inv.arguments,
                        "success": inv.success,
                        "latency_ms": inv.latency_ms,
                        "mcp_call": True,
                        "result": inv.result,
                    })
                    messages.append(
                        ToolExecutor.tool_message(
                            tc["id"], inv.result if inv.result else {"error": "no result"}
                        )
                    )

                    # If the LLM called RAG itself, prefer its query's results.
                    if (
                        inv.tool == "search_policy_documents"
                        and inv.success
                        and inv.result
                    ):
                        llm_chunks = self._chunks_from_tool_result(inv.result)
                        if llm_chunks:
                            retrieved_chunks = llm_chunks
                            rag_source = "llm_called"
                            trace.append({
                                "step": "rag_llm_override",
                                "timestamp": inv.started_at,
                                "chunks_retrieved": len(retrieved_chunks),
                                "scores": [c.get("score", 0) for c in retrieved_chunks],
                            })

        if final_answer is None:
            # Loop exhausted without a textual answer; force one more call
            # with tool_choice="none" so the model must produce a reply.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "You have already gathered all available tool data. "
                            "Now respond to the user in plain text only. "
                            "Do NOT emit any tool-call XML or pseudo-XML."
                        ),
                    },
                ],
                tool_choice="none",
                temperature=0.3,
                max_tokens=4096,
            )
            final_answer = response.choices[0].message.content or ""
            final_answer = self._strip_tool_call_artifacts(final_answer)
            trace.append({
                "step": "final_answer_forced",
                "timestamp": datetime.now().isoformat(),
                "reason": "max_iterations_reached",
            })

        # ----- tool guard: force any missing mandatory tools ---------
        # Skip when rag_only=True (pure policy question — no employee tools needed).
        forced = [] if rag_only else await self._enforce_tool_guard(query, employee_id, invocations)
        invocations.extend(forced)
        for inv in forced:
            trace.append({
                "step": "tool_guard_executed",
                "timestamp": inv.started_at,
                "tool": inv.tool,
                "arguments": inv.arguments,
                "success": inv.success,
                "latency_ms": inv.latency_ms,
                "mcp_call": True,
                "result": inv.result,
                "reason": "tool_guard",
            })

        # ----- RAG retrieval for synthesizer -------------------------
        # Note: ``retrieved_chunks`` is now populated by Stage 0.5 (pre-loop
        # retrieval, cached) and optionally overridden if the LLM called
        # ``search_policy_documents`` itself in Stage 1. We only re-retrieve
        # here if neither path produced any chunks.
        if not retrieved_chunks and self.rag_pipeline:
            retrieved_chunks = self._retrieve_rag_once(query, k=5)
            if retrieved_chunks:
                rag_source = "fallback_post_loop"
        trace.append({
            "step": "rag_final",
            "timestamp": datetime.now().isoformat(),
            "source": rag_source,
            "chunks_retrieved": len(retrieved_chunks),
            "scores": [c.get("score", 0) for c in retrieved_chunks],
        })

        # ----- final synthesis with tool + RAG context --------------
        employee_context = self._extract_employee_context(invocations)

        # Detect "employee not found" cases from MCP tool results and short-circuit
        # before invoking the synthesizer (avoids LLM hallucinating a PTO balance
        # for an ID that does not exist in the data store).
        not_found = self._detect_employee_not_found(invocations, employee_id)
        if not_found:
            final_answer = (
                f"I could not find employee **{employee_id}** in our HR system. "
                "Please double-check the employee ID (it should look like 'EMP001'). "
                "If you are sure the ID is correct, contact HR directly for help."
            )
            trace.append({
                "step": "synthesis_short_circuit",
                "timestamp": datetime.now().isoformat(),
                "reason": "employee_not_found",
                "employee_id": employee_id,
            })
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            return {
                "answer": final_answer,
                "citations": [],
                "sources_used": [],
                "tool_calls": [inv.to_dict() for inv in invocations],
                "trace": trace,
"metadata": {
                "latency_ms": latency_ms,
                "model": self.model,
                "chunks_retrieved": 0,
                "tools_used": len(invocations),
                "mcp_used": bool(invocations),
                "mcp_connected": mcp_connected,
                "loop_iterations": loop_steps,
                "rag_source": rag_source,
                "timestamp": datetime.now().isoformat(),
            },
        }

        final_prompt = self.planner.build_user_prompt(
            query=query,
            tool_results=[inv.to_dict() for inv in invocations],
            retrieved_chunks=retrieved_chunks,
            employee_context=employee_context,
        )
        synth_started = time.time()
        synth = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        TaskPlanner.RAG_ONLY_SYSTEM_PROMPT
                        if rag_only
                        else TaskPlanner.SYSTEM_PROMPT
                    ),
                },
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        raw_synth = synth.choices[0].message.content or ""
        final_answer = self._strip_tool_call_artifacts(raw_synth) or final_answer
        trace.append({
            "step": "synthesis",
            "timestamp": datetime.now().isoformat(),
            "latency_ms": int((time.time() - synth_started) * 1000),
        })

        # Add source summary to answer if sources were referenced
        sources_in_answer = set()
        for i in range(1, len(retrieved_chunks) + 1):
            if f"[Source {i}]" in final_answer:
                sources_in_answer.add(i)

        if sources_in_answer:
            # Build source mapping with section info
            source_map = {}
            for i, c in enumerate(retrieved_chunks, 1):
                chunk_meta = c.get("metadata", {}) or {}
                filename = chunk_meta.get("filename") or c.get("document_id", "unknown")
                # Use the primary heading (first section) as the section reference
                heading = c.get("heading", "") or "General"
                # Take only the first heading if multiple are joined
                primary_heading = heading.split(" / ")[0].strip() if heading else "General"
                source_map[i] = (filename, primary_heading)

            # Add clear source mapping at the end (no hyperlinks)
            summary_lines = ["\n\n---\n**References:**"]
            for i in sorted(sources_in_answer):
                filename, heading = source_map.get(i, ("unknown", ""))
                summary_lines.append(f"- Source {i}: {filename} — {heading}")
            final_answer += "\n".join(summary_lines)

        # Build citations with proper filename handling. Only include
        # sources that the model actually referenced in its answer (so the
        # UI doesn't show extra, unrelated sources like leave-policy.md
        # when the response only cited pto-policy.md).
        citations = []
        for i, c in enumerate(retrieved_chunks, 1):
            if i not in sources_in_answer:
                continue
            chunk_meta = c.get("metadata") or {}
            # Get filename from metadata, or construct from document_id with correct extension
            filename = chunk_meta.get("filename")
            if not filename:
                # Try to detect format from metadata
                fmt = chunk_meta.get("format", ".md")
                filename = f"{c.get('document_id', 'unknown')}{fmt}"

            citations.append({
                "source_number": i,
                "document_id": c.get("document_id"),
                "title": c.get("title"),
                "heading": c.get("heading", ""),
                "source": chunk_meta.get("source", ""),
                "filename": filename,
                "snippet": (c.get("content") or "")[:500],
            })

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        mcp_used = any(inv.tool != "search_policy_documents" for inv in invocations)

        return {
            "answer": final_answer,
            "citations": citations,
            "sources_used": [c.get("document_id") for c in retrieved_chunks],
            "tool_calls": [inv.to_dict() for inv in invocations],
            "trace": trace,
            "metadata": {
                "latency_ms": latency_ms,
                "model": self.model,
                "chunks_retrieved": len(retrieved_chunks),
                "tools_used": len(invocations),
                "mcp_used": mcp_used or bool(invocations),
                "mcp_connected": mcp_connected,
                "loop_iterations": loop_steps,
                "rag_source": rag_source,
                "timestamp": datetime.now().isoformat(),
            },
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _build_user_turn(self, query: str, employee_id: Optional[str]) -> str:
        """First user message; surfaces known employee_id so the LLM uses it."""
        if employee_id:
            return (
                f"Employee ID: {employee_id}\n"
                f"Question: {query}\n\n"
                "Use the available MCP tools to gather accurate information, "
                "ground policy claims via search_policy_documents, and "
                "cite sources using [Source N] notation."
            )
        return (
            f"Question: {query}\n\n"
            "Use the available MCP tools to gather accurate information, "
            "ground policy claims via search_policy_documents, and "
            "cite sources using [Source N] notation."
        )

    @staticmethod
    def _extract_employee_context(invocations: List[ToolInvocation]) -> Dict[str, Any]:
        for inv in invocations:
            if (
                inv.tool == "lookup_employee_profile"
                and inv.result
                and inv.result.get("success")
            ):
                return inv.result.get("employee") or {}
        return {}

    @staticmethod
    def _format_rag_context(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return ""
        blocks = []
        for i, c in enumerate(chunks, 1):
            blocks.append(
                f"[Source {i}: {c.get('title', 'Unknown')}]\n"
                f"Section: {c.get('heading', 'N/A')}\n"
                f"{c.get('content', '')}"
            )
        return "Policy Documents (from RAG):\n\n" + "\n\n---\n\n".join(blocks)

    @staticmethod
    def _chunks_from_tool_result(
        result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Reshape ``search_policy_documents`` MCP result into retriever-shaped chunks.

        The MCP tool returns ``{"results": [{"document_id": ..., "title": ...,
        "heading": ..., "content": ..., "score": ...}, ...]}``; the rest of
        the pipeline expects the retriever's raw chunk dict. This adapter
        lets us treat the LLM-called retrieval and the pre-loop retrieval
        uniformly.
        """
        if not isinstance(result, dict):
            return []
        items = result.get("results") or []
        chunks: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            source = item.get("source", "")
            chunks.append({
                "document_id": item.get("document_id"),
                "title": item.get("title"),
                "heading": item.get("heading", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
                "metadata": {"source": source} if source else {},
            })
        return chunks

    @staticmethod
    def _strip_tool_call_artifacts(text: str) -> str:
        """Strip DSML/pseudo tool-call XML that some models leak into ``content``.

        DeepSeek's flash variants occasionally emit ``<|DSML|>tool_calls>...``
        inside the assistant ``content`` field. We strip the entire block if
        present, returning whatever plain-text reply remains.
        """
        if not text:
            return ""
        # Cover both ASCII (real tool call header) and fullwidth / spaced
        # variants some models emit in the assistant text stream.
        markers = (
            "<|DSML|>", "<|dsml|>",
            "<|python_tag|>", "<|python_tag",
            "<|tool_call|>", "<|tool_call",
            "<|tool_call_begin|>", "<|tool_calls_section_end|>",
            "｜DSML｜", "｜dsml｜",
            "<tool_call>", "<tool_calls>",
        )
        if not any(m in text for m in markers):
            return text.strip()
        # Find the earliest start-marker and cut there.
        cut_at = len(text)
        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                cut_at = min(cut_at, idx)
        return text[:cut_at].strip()

    @staticmethod
    def _detect_employee_not_found(
        invocations: List[ToolInvocation], employee_id: Optional[str]
    ) -> bool:
        """True if the profile/PTO/benefits tools all reported the employee is missing."""
        if not employee_id:
            return False

        lookup_tools = {
            "lookup_employee_profile",
            "check_pto_balance",
            "lookup_benefits_status",
            "check_policy_compliance",
        }
        relevant = [inv for inv in invocations if inv.tool in lookup_tools]
        if not relevant:
            return False

        def _is_not_found(inv: ToolInvocation) -> bool:
            result = inv.result or {}
            if result.get("success"):
                return False
            err = (result.get("error") or "").lower()
            return ("not found" in err) or ("employee_not_found" in err)

        return all(_is_not_found(inv) for inv in relevant)

    async def _enforce_tool_guard(
        self,
        query: str,
        employee_id: Optional[str],
        invocations: List[ToolInvocation],
    ) -> List[ToolInvocation]:
        """Force-call any mandatory tool the LLM skipped.

        Skipping is detected by inspecting the in-loop ``invocations``
        list for tool names that already ran; missing ones are issued
        in parallel and appended so the synthesizer sees them.
        """
        if not employee_id or not self.executor.mcp_client:
            return []

        called = {inv.tool for inv in invocations}

        required = self._mandatory_tools(query)
        missing = [t for t in required if t not in called]
        if not missing:
            return []

        # Filter to tools whose required args we can fill from employee_id.
        payload: List[Dict[str, Any]] = []
        for tool in missing:
            args = self._default_args_for(tool, employee_id)
            if args is None:
                continue
            payload.append({"name": tool, "arguments": args})

        if not payload:
            return []

        return await self.executor.call_many(payload)

    def _mandatory_tools(self, query: str) -> List[str]:
        """Map a query to mandatory MCP tool names."""
        if not query:
            return []
        tools: List[str] = []
        q = query.lower()
        if any(kw in q for kw in TaskPlanner.PTO_KEYWORDS):
            tools.append("check_pto_balance")
        if any(kw in q for kw in TaskPlanner.BENEFITS_KEYWORDS):
            tools.append("lookup_benefits_status")
        if any(kw in q for kw in TaskPlanner.REMOTE_KEYWORDS):
            tools.append("lookup_employee_profile")
            tools.append("check_policy_compliance")
        if any(kw in q for kw in TaskPlanner.PROFILE_KEYWORDS):
            tools.append("lookup_employee_profile")
        # de-dup preserve order
        seen: set = set()
        return [t for t in tools if not (t in seen or seen.add(t))]

    @staticmethod
    def _default_args_for(tool: str, employee_id: str) -> Optional[Dict[str, Any]]:
        defaults: Dict[str, Dict[str, Any]] = {
            "lookup_employee_profile": {"employee_id": employee_id},
            "check_pto_balance": {"employee_id": employee_id, "year": 2026},
            "lookup_benefits_status": {"employee_id": employee_id},
            "check_policy_compliance": {
                "employee_id": employee_id,
                "policy_area": "remote_work",
            },
        }
        return defaults.get(tool)

    # ------------------------------------------------------------------ #
    # Misc                                                                #
    # ------------------------------------------------------------------ #

    def process_request_sync(
        self,
        query: str,
        employee_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.process_request(query, employee_id)
        )

    def get_capabilities(self) -> Dict[str, Any]:
        tools = (
            [t["function"]["name"] for t in (self.executor._openai_tools_cache or [])]
            if self.executor and self.executor._openai_tools_cache
            else [
                "lookup_employee_profile",
                "check_pto_balance",
                "lookup_benefits_status",
                "create_mock_hr_ticket",
                "draft_hr_email",
                "check_policy_compliance",
                "search_policy_documents",
                "get_policy_section",
            ]
        )
        return {
            "tools": tools,
            "mcp_server": self.mcp_server_url,
            "mcp_protocol_used": self.use_mcp_protocol,
            "model": self.model,
            "rag_enabled": self.rag_pipeline is not None,
        }
