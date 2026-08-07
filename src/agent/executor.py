"""Agent executor: runs the tool-calling loop and resolves dependencies.

This module is the agent's "executor" half (the planner sits in
``planner.py``). It is responsible for:

1. Loading MCP tool definitions and converting them into OpenAI
   ``tools`` schema so the LLM can call them via native tool calling.
2. Driving a multi-turn conversation loop where the LLM can issue tool
   calls, we execute them in parallel via MCP, and feed the results
   back as ``role: tool`` messages until the model returns a final
   answer.
3. Honouring a hard maximum on tool-call iterations and surfacing
   latency/token information in the trace.

Native tool calling is preferred over the previous JSON-text planner
because it eliminates the "model says it will check PTO but the plan
does not list check_pto_balance" failure mode - the model either
emits a structured tool call or directly returns the answer.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional


ToolCallFn = Callable[[str, Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]


@dataclass
class ToolInvocation:
    """A single recorded tool invocation."""

    tool: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    success: bool
    latency_ms: int
    mcp_call: bool = True
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "result": self.result,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "mcp_call": self.mcp_call,
            "started_at": self.started_at,
        }


class ToolExecutor:
    """Executes MCP tool calls, optionally in parallel batches.

    The executor is *the* place where tool selection is enforced. The
    orchestrator delegates tool routing here so that:

    * tool descriptions live in one place (loaded from MCP server);
    * parallel independent calls reduce P95 latency;
    * latency and success metrics are recorded consistently.
    """

    def __init__(
        self,
        mcp_client,
        tool_call_fn: Optional[ToolCallFn] = None,
        max_parallel: int = 4,
    ):
        self.mcp_client = mcp_client
        # Allow dependency injection for tests.
        self._tool_call_fn = tool_call_fn
        self.max_parallel = max_parallel
        self._openai_tools_cache: Optional[List[Dict[str, Any]]] = None

    async def load_openai_tools(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch MCP tool list and adapt to OpenAI function-calling schema."""
        if self._openai_tools_cache and not refresh:
            return self._openai_tools_cache

        tools = await self.mcp_client.list_tools()
        adapted: List[Dict[str, Any]] = []
        for t in tools:
            adapted.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("inputSchema") or {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            )
        self._openai_tools_cache = adapted
        return adapted

    async def call_one(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolInvocation:
        """Call a single tool via MCP and capture timing/result."""
        started = time.time()
        if self._tool_call_fn is not None:
            result = await self._tool_call_fn(tool_name, arguments)
        else:
            result = await self.mcp_client.call_tool(tool_name, arguments)
        latency_ms = int((time.time() - started) * 1000)

        success = bool(result and result.get("success"))
        return ToolInvocation(
            tool=tool_name,
            arguments=arguments,
            result=result,
            success=success,
            latency_ms=latency_ms,
        )

    async def call_many(
        self, calls: List[Dict[str, Any]]
    ) -> List[ToolInvocation]:
        """Execute multiple independent tool calls in parallel.

        ``calls`` is a list of ``{"name": str, "arguments": dict}``.
        """
        if not calls:
            return []
        sem = asyncio.Semaphore(self.max_parallel)

        async def _run(call: Dict[str, Any]) -> ToolInvocation:
            async with sem:
                return await self.call_one(
                    call["name"], call.get("arguments") or {}
                )

        return await asyncio.gather(*[_run(c) for c in calls])

    @staticmethod
    def parse_tool_calls(message) -> List[Dict[str, Any]]:
        """Extract normalised tool-call dicts from an OpenAI chat message.

        Returns ``[{"id": ..., "name": ..., "arguments": {...}}]``.
        Tolerates both the OpenAI SDK object shape and a plain dict for
        testability.
        """
        raw = getattr(message, "tool_calls", None)
        if raw is None and isinstance(message, dict):
            raw = message.get("tool_calls")
        if not raw:
            return []

        out: List[Dict[str, Any]] = []
        for tc in raw:
            # SDK object: tc.id, tc.function.name, tc.function.arguments
            tc_id = getattr(tc, "id", None) or tc.get("id")
            fn = getattr(tc, "function", None) or tc.get("function") or {}
            name = getattr(fn, "name", None) or fn.get("name")
            args_raw = getattr(fn, "arguments", None) or fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            out.append({"id": tc_id, "name": name, "arguments": args})
        return out

    @staticmethod
    def assistant_tool_calls_message(message) -> Dict[str, Any]:
        """Convert an assistant message with tool_calls to a serialisable dict.

        Required when appending the assistant turn to the messages list
        before tool results, since subsequent calls must see the
        original tool_calls payload verbatim.
        """
        if isinstance(message, dict):
            return message
        return {
            "role": "assistant",
            "content": getattr(message, "content", "") or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (getattr(message, "tool_calls", None) or [])
            ],
        }

    @staticmethod
    def tool_message(call_id: str, content: Any) -> Dict[str, Any]:
        """Build a ``role: tool`` message for the LLM."""
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False, default=str)
            except TypeError:
                content = str(content)
        return {"role": "tool", "tool_call_id": call_id, "content": content}
