"""Tests for MCP integration."""

import pytest


class TestMCPToolsIntegration:
    """Test MCP tools integration."""

    @pytest.mark.asyncio
    async def test_all_tools_registered(self):
        """Test that all required tools are registered."""
        from src.mcp_server.app import create_app

        # ``create_app`` returns the FastMCP instance with all tools
        # registered. Previously the test imported a non-existent
        # ``mcp_server`` symbol — the module exposes a factory
        # (``create_app`` / ``get_mcp_server``) instead.
        mcp_server = create_app()
        # ``FastMCP.list_tools`` is async; with ``asyncio_mode = "auto"``
        # in pyproject.toml, this test is automatically wrapped and run
        # by pytest-asyncio. Calling ``get_event_loop().run_until_complete``
        # from inside a sync test would deadlock, so we await directly.
        tools = await mcp_server.list_tools()

        # FastMCP 3.x returns ``FunctionTool`` Pydantic models that
        # expose the tool name as ``.name`` — but tolerate a plain dict
        # shape too so the test is robust against upstream churn.
        tool_names = []
        for t in tools:
            if isinstance(t, dict):
                tool_names.append(t.get("name"))
            else:
                tool_names.append(getattr(t, "name", None))

        required_tools = [
            "lookup_employee_profile",
            "check_pto_balance",
            "lookup_benefits_status",
            "create_mock_hr_ticket",
            "draft_hr_email",
            "check_policy_compliance",
            "search_policy_documents",
            "get_policy_section",
        ]

        missing = [t for t in required_tools if t not in tool_names]
        assert not missing, f"Missing MCP tools: {missing} (got {tool_names})"

    def test_tool_call_workflow(self):
        """Test a complete tool call workflow."""
        from src.mcp_server.tools import HRTools

        tools = HRTools(data_dir="mock_data")

        # Step 1: Look up employee
        emp_result = tools.lookup_employee_profile(employee_id="EMP001")
        assert emp_result["success"] is True

        # Step 2: Check PTO balance
        pto_result = tools.check_pto_balance(employee_id="EMP001")
        assert pto_result["success"] is True

        # Step 3: Check benefits
        benefits_result = tools.lookup_benefits_status(employee_id="EMP001")
        assert benefits_result["success"] is True

        # Step 4: Create ticket
        ticket_result = tools.create_mock_hr_ticket(
            employee_id="EMP001",
            category="pto",
            subject="PTO Request",
            description="Requesting time off"
        )
        assert ticket_result["success"] is True

        # Step 5: Draft email
        email_result = tools.draft_hr_email(
            employee_id="EMP001",
            purpose="pto_request"
        )
        assert email_result["success"] is True

    def test_complex_workflow(self):
        """Test complex multi-step workflow."""
        from src.mcp_server.tools import HRTools

        tools = HRTools(data_dir="mock_data")

        # Remote work eligibility check workflow
        # Step 1: Get employee profile
        emp = tools.lookup_employee_profile(employee_id="EMP002")
        assert emp["success"] is True
        assert emp["employee"]["work_arrangement"] == "remote"

        # Step 2: Check compliance
        compliance = tools.check_policy_compliance(
            employee_id="EMP002",
            policy_area="remote_work"
        )
        # check_policy_compliance returns compliance status directly
        assert "compliant" in compliance
        assert compliance["compliant"] is True

        # Step 3: Create ticket for tracking
        ticket = tools.create_mock_hr_ticket(
            employee_id="EMP002",
            category="remote_work",
            subject="Remote Work Request",
            description="Employee working remotely"
        )
        assert ticket["success"] is True

        # Step 4: Draft confirmation email
        email = tools.draft_hr_email(
            employee_id="EMP002",
            purpose="remote_work_approval"
        )
        assert email["success"] is True
        assert "remote" in email["email"]["subject"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
