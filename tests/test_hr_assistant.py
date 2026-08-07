"""Test cases for HR Policy Assistant.

Note: These tests focus on the tool layer. Full orchestrator tests require
a running API server with RAG pipeline initialized.
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcp.tools import HRTools


class TestEmployeeProfile:
    """Test employee profile lookups."""

    @pytest.fixture
    def hr_tools(self):
        return HRTools()

    def test_lookup_emp001(self, hr_tools):
        """EMP001 should be Alice Johnson, Engineering."""
        result = hr_tools.lookup_employee_profile(employee_id="EMP001")

        assert result["success"] is True
        emp = result["employee"]
        assert emp["name"] == "Alice Johnson"
        assert emp["department"] == "Engineering"
        assert emp["title"] == "Senior Software Engineer"

    def test_lookup_emp005(self, hr_tools):
        """EMP005 should be Emma Brown."""
        result = hr_tools.lookup_employee_profile(employee_id="EMP005")

        assert result["success"] is True
        emp = result["employee"]
        assert emp["name"] == "Emma Brown"
        assert emp["employee_id"] == "EMP005"

    def test_lookup_nonexistent_employee(self, hr_tools):
        """Non-existent employee should return error."""
        result = hr_tools.lookup_employee_profile(employee_id="EMP999")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestPTOBalanceTool:
    """Test PTO balance tool directly."""

    @pytest.fixture
    def hr_tools(self):
        return HRTools()

    def test_emp001_pto(self, hr_tools):
        """EMP001 should have 8 available days (15 accrued - 5 used - 2 pending)."""
        result = hr_tools.check_pto_balance(employee_id="EMP001")

        assert result["success"] is True
        assert result["available_days"] == 8
        assert result["accrued_days"] == 15
        assert result["used_days"] == 5

    def test_emp005_pto(self, hr_tools):
        """EMP005 should have 3 available days (15 accrued - 7 used - 5 pending = 3)."""
        result = hr_tools.check_pto_balance(employee_id="EMP005")

        assert result["success"] is True
        assert result["available_days"] == 3

    def test_emp002_pto(self, hr_tools):
        """EMP002 should have 12 available days (15 accrued - 3 used)."""
        result = hr_tools.check_pto_balance(employee_id="EMP002")

        assert result["success"] is True
        assert result["available_days"] == 12
        assert result["accrued_days"] == 15


class TestBenefitsStatus:
    """Test benefits status queries."""

    @pytest.fixture
    def hr_tools(self):
        return HRTools()

    def test_emp001_benefits(self, hr_tools):
        """EMP001 should have active benefits."""
        result = hr_tools.lookup_benefits_status(employee_id="EMP001")

        assert result["success"] is True
        assert "medical" in result
        assert "dental_enrolled" in result

    def test_emp003_benefits(self, hr_tools):
        """EMP003 should have benefits data."""
        result = hr_tools.lookup_benefits_status(employee_id="EMP003")

        assert result["success"] is True
        assert "medical" in result


class TestHRTickets:
    """Test HR ticket creation."""

    @pytest.fixture
    def hr_tools(self):
        return HRTools()

    def test_create_ticket(self, hr_tools):
        """Should be able to create an HR ticket."""
        result = hr_tools.create_mock_hr_ticket(
            employee_id="EMP001",
            category="pto",  # lowercase
            subject="Test ticket",
            description="Test description"
        )

        assert result["success"] is True
        assert "ticket" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
