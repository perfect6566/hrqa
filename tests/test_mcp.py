"""Tests for MCP integration."""

import pytest


class TestMCPToolsIntegration:
    """Test MCP tools integration."""

    def test_all_tools_registered(self):
        """Test that all required tools are registered."""
        from src.mcp.app import mcp_server

        tools = mcp_server.list_tools()
        tool_names = [t["name"] for t in tools]

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

        for tool in required_tools:
            assert tool in tool_names, f"Missing tool: {tool}"

    def test_tool_call_workflow(self):
        """Test a complete tool call workflow."""
        from src.mcp.tools import HRTools

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
        from src.mcp.tools import HRTools

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
