"""Unit tests for ``src.agent.executor.ToolExecutor``.

The executor is the orchestrator's tool-routing layer:

* loads MCP tool definitions and adapts them to OpenAI's function-calling schema;
* executes a single tool call and records ``ToolInvocation`` metrics;
* runs multiple calls in parallel via ``call_many``;
* parses / re-serialises tool-call payloads between SDK objects and dicts.

We stub the MCP client with ``FakeMCPClient`` from ``conftest.py`` so
no network calls happen, and we use a synchronous ``_tool_call_fn``
where convenient to keep the assertions easy to read.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.agent.executor import ToolExecutor, ToolInvocation

from tests.conftest import FakeMCPClient


# --------------------------------------------------------------------- #
# dataclass                                                             #
# --------------------------------------------------------------------- #

class TestToolInvocation:
    """The plain dataclass that records a single tool call."""

    def test_to_dict_roundtrip(self) -> None:
        inv = ToolInvocation(
            tool="lookup_employee_profile",
            arguments={"employee_id": "EMP001"},
            result={"success": True, "employee": {"name": "Alice"}},
            success=True,
            latency_ms=12,
        )
        d = inv.to_dict()
        assert d["tool"] == "lookup_employee_profile"
        assert d["arguments"] == {"employee_id": "EMP001"}
        assert d["success"] is True
        assert d["latency_ms"] == 12

    def test_default_started_at_is_iso_string(self) -> None:
        inv = ToolInvocation(
            tool="x",
            arguments={},
            result=None,
            success=False,
            latency_ms=0,
        )
        # Just assert it is a non-empty string; we don't pin the exact time.
        assert isinstance(inv.started_at, str)
        assert inv.started_at


# --------------------------------------------------------------------- #
# load_openai_tools                                                     #
# --------------------------------------------------------------------- #

class TestLoadOpenaiTools:
    """Adapt MCP tool schemas into OpenAI's function-calling shape."""

    @pytest.mark.asyncio
    async def test_adapts_mcp_schema_to_openai_shape(self) -> None:
        mcp_tools = [
            {
                "name": "check_pto_balance",
                "description": "Check PTO balance",
                "inputSchema": {
                    "type": "object",
                    "properties": {"employee_id": {"type": "string"}},
                },
            },
        ]
        mcp = FakeMCPClient(tools=mcp_tools)
        executor = ToolExecutor(mcp_client=mcp)

        adapted = await executor.load_openai_tools()

        assert adapted == [
            {
                "type": "function",
                "function": {
                    "name": "check_pto_balance",
                    "description": "Check PTO balance",
                    "parameters": {
                        "type": "object",
                        "properties": {"employee_id": {"type": "string"}},
                    },
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_caches_tools_between_calls(self) -> None:
        mcp = FakeMCPClient(
            tools=[{"name": "t1", "description": "d", "inputSchema": {}}]
        )
        executor = ToolExecutor(mcp_client=mcp)

        first = await executor.load_openai_tools()
        # Mutate the MCP tool list so we can prove caching kicked in.
        mcp._tools = [{"name": "different", "description": "d", "inputSchema": {}}]
        second = await executor.load_openai_tools()
        assert first == second

    @pytest.mark.asyncio
    async def test_refresh_force_reloads(self) -> None:
        mcp = FakeMCPClient(
            tools=[{"name": "t1", "description": "d", "inputSchema": {}}]
        )
        executor = ToolExecutor(mcp_client=mcp)

        await executor.load_openai_tools()
        mcp._tools = [{"name": "t2", "description": "d", "inputSchema": {}}]
        reloaded = await executor.load_openai_tools(refresh=True)
        assert reloaded[0]["function"]["name"] == "t2"

    @pytest.mark.asyncio
    async def test_missing_description_defaults_to_empty_string(self) -> None:
        mcp = FakeMCPClient(tools=[{"name": "t1", "inputSchema": {}}])
        executor = ToolExecutor(mcp_client=mcp)
        adapted = await executor.load_openai_tools()
        assert adapted[0]["function"]["description"] == ""

    @pytest.mark.asyncio
    async def test_missing_input_schema_defaults_to_empty_object(self) -> None:
        mcp = FakeMCPClient(tools=[{"name": "t1", "description": "d"}])
        executor = ToolExecutor(mcp_client=mcp)
        adapted = await executor.load_openai_tools()
        assert adapted[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }


# --------------------------------------------------------------------- #
# call_one                                                              #
# --------------------------------------------------------------------- #

class TestCallOne:
    """Execute a single MCP tool call and record timing."""

    @pytest.mark.asyncio
    async def test_records_success(self) -> None:
        mcp = FakeMCPClient(
            call_results={
                "check_pto_balance": {"success": True, "available_days": 8},
            }
        )
        executor = ToolExecutor(mcp_client=mcp)
        inv = await executor.call_one(
            "check_pto_balance", {"employee_id": "EMP001"}
        )
        assert inv.tool == "check_pto_balance"
        assert inv.arguments == {"employee_id": "EMP001"}
        assert inv.success is True
        assert inv.result == {"success": True, "available_days": 8}
        assert inv.latency_ms >= 0
        assert inv.mcp_call is True

    @pytest.mark.asyncio
    async def test_records_failure(self) -> None:
        mcp = FakeMCPClient(
            call_results={
                "check_pto_balance": {"success": False, "error": "not found"},
            }
        )
        executor = ToolExecutor(mcp_client=mcp)
        inv = await executor.call_one("check_pto_balance", {"employee_id": "EMP999"})
        assert inv.success is False
        assert inv.result["error"] == "not found"

    @pytest.mark.asyncio
    async def test_injects_tool_call_fn_when_set(self) -> None:
        """When ``_tool_call_fn`` is provided, MCP is bypassed."""
        calls: list = []

        async def fake_call(name, arguments):
            calls.append((name, arguments))
            return {"success": True, "echoed": arguments}

        executor = ToolExecutor(mcp_client=None, tool_call_fn=fake_call)
        inv = await executor.call_one("foo", {"x": 1})
        assert inv.success is True
        assert inv.result == {"success": True, "echoed": {"x": 1}}
        assert calls == [("foo", {"x": 1})]


# --------------------------------------------------------------------- #
# call_many                                                             #
# --------------------------------------------------------------------- #

class TestCallMany:
    """Run independent tool calls in parallel."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self) -> None:
        mcp = FakeMCPClient()
        executor = ToolExecutor(mcp_client=mcp)
        assert await executor.call_many([]) == []

    @pytest.mark.asyncio
    async def test_runs_all_calls(self) -> None:
        mcp = FakeMCPClient(
            call_results={
                "a": {"success": True, "v": 1},
                "b": {"success": True, "v": 2},
            }
        )
        executor = ToolExecutor(mcp_client=mcp, max_parallel=4)
        results = await executor.call_many(
            [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]
        )
        assert len(results) == 2
        assert {r.tool for r in results} == {"a", "b"}
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_missing_arguments_defaults_to_empty_dict(self) -> None:
        mcp = FakeMCPClient(
            call_results={"a": {"success": True, "v": 1}},
        )
        executor = ToolExecutor(mcp_client=mcp)
        # No "arguments" key in the payload.
        results = await executor.call_many([{"name": "a"}])
        assert results[0].arguments == {}


# --------------------------------------------------------------------- #
# parse_tool_calls                                                      #
# --------------------------------------------------------------------- #

class TestParseToolCalls:
    """Extract normalised tool-call dicts from OpenAI chat messages."""

    def test_returns_empty_when_no_tool_calls(self) -> None:
        # Plain dict message with no tool_calls.
        msg = {"role": "assistant", "content": "hello"}
        assert ToolExecutor.parse_tool_calls(msg) == []

    def test_parses_dict_message(self) -> None:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "check_pto_balance",
                        "arguments": '{"employee_id": "EMP001"}',
                    },
                }
            ],
        }
        out = ToolExecutor.parse_tool_calls(msg)
        assert out == [
            {
                "id": "call_1",
                "name": "check_pto_balance",
                "arguments": {"employee_id": "EMP001"},
            }
        ]

    def test_parses_sdk_message(self) -> None:
        class FakeFn:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class FakeTC:
            def __init__(self, id, fn):
                self.id = id
                self.function = fn

        class FakeMsg:
            def __init__(self):
                self.tool_calls = [
                    FakeTC("call_2", FakeFn("foo", '{"a": 1}')),
                ]

        out = ToolExecutor.parse_tool_calls(FakeMsg())
        assert out == [{"id": "call_2", "name": "foo", "arguments": {"a": 1}}]

    def test_invalid_json_arguments_default_to_empty_dict(self) -> None:
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_3",
                    "function": {
                        "name": "foo",
                        "arguments": "not-json{",
                    },
                }
            ],
        }
        out = ToolExecutor.parse_tool_calls(msg)
        assert out == [{"id": "call_3", "name": "foo", "arguments": {}}]


# --------------------------------------------------------------------- #
# assistant_tool_calls_message / tool_message                            #
# --------------------------------------------------------------------- #

class TestMessageBuilders:
    """Build the serialisable messages the orchestrator appends to history."""

    def test_assistant_tool_calls_message_passes_dict_through(self) -> None:
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        }
        assert ToolExecutor.assistant_tool_calls_message(msg) is msg

    def test_assistant_tool_calls_message_serialises_sdk(self) -> None:
        class FakeFn:
            def __init__(self):
                self.name = "check_pto_balance"
                self.arguments = '{"employee_id": "EMP001"}'

        class FakeTC:
            def __init__(self):
                self.id = "call_1"
                self.function = FakeFn()

        class FakeMsg:
            content = ""
            tool_calls = [FakeTC()]

        out = ToolExecutor.assistant_tool_calls_message(FakeMsg())
        assert out["role"] == "assistant"
        assert out["content"] == ""
        assert out["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "check_pto_balance",
                    "arguments": '{"employee_id": "EMP001"}',
                },
            }
        ]

    def test_tool_message_stringifies_non_string_content(self) -> None:
        msg = ToolExecutor.tool_message("call_1", {"success": True, "v": 1})
        assert msg == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps({"success": True, "v": 1}),
        }

    def test_tool_message_passes_through_strings(self) -> None:
        msg = ToolExecutor.tool_message("call_1", "raw text")
        assert msg == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "raw text",
        }


# --------------------------------------------------------------------- #
# Smoke                                                                 #
# --------------------------------------------------------------------- #

def test_executor_smoke_runs_event_loop() -> None:
    """A trivial async smoke test that proves asyncio + executor glue works."""
    mcp = FakeMCPClient(call_results={"x": {"success": True}})

    async def go() -> ToolInvocation:
        executor = ToolExecutor(mcp_client=mcp)
        return await executor.call_one("x", {})

    inv = asyncio.run(go())
    assert inv.success is True
