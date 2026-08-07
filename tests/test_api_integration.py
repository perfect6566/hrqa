"""Integration tests via API client.

These tests require a running API server. Run with:
    pytest tests/test_api_integration.py -v
"""

import pytest
import httpx


BASE_URL = "http://localhost:8008"


@pytest.fixture
def api_client():
    """HTTP client for API testing."""
    return httpx.Client(base_url=BASE_URL, timeout=60.0)


class TestPTOBalanceAPI:
    """Test PTO balance via API."""

    def test_emp001_pto(self, api_client):
        """EMP001 should have 8 days available - verify via tool_calls."""
        response = api_client.post(
            "/chat",
            json={
                "message": "How many PTO days do I have?",
                "employee_id": "EMP001"
            }
        )
        assert response.status_code == 200
        data = response.json()
        # Verify PTO balance is returned via tool
        tool_calls = data.get("tool_calls", [])
        pto_call = next((tc for tc in tool_calls if tc.get("tool") == "check_pto_balance"), None)
        assert pto_call is not None, "PTO balance tool should be called"
        pto_result = pto_call.get("result", {}).get("result", {})
        assert pto_result.get("available_days") == 8, f"Expected 8 days, got {pto_result}"

    def test_emp005_pto(self, api_client):
        """EMP005 should have 3 days available - verify via tool_calls."""
        response = api_client.post(
            "/chat",
            json={
                "message": "What is my PTO balance?",
                "employee_id": "EMP005"
            }
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        pto_call = next((tc for tc in tool_calls if tc.get("tool") == "check_pto_balance"), None)
        assert pto_call is not None, "PTO balance tool should be called"
        pto_result = pto_call.get("result", {}).get("result", {})
        assert pto_result.get("available_days") == 3, f"Expected 3 days, got {pto_result}"

    def test_emp002_pto(self, api_client):
        """EMP002 should have 12 days available - verify via tool_calls."""
        response = api_client.post(
            "/chat",
            json={
                "message": "How many vacation days do I have left?",
                "employee_id": "EMP002"
            }
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        pto_call = next((tc for tc in tool_calls if tc.get("tool") == "check_pto_balance"), None)
        assert pto_call is not None, "PTO balance tool should be called"
        pto_result = pto_call.get("result", {}).get("result", {})
        assert pto_result.get("available_days") == 12, f"Expected 12 days, got {pto_result}"


class TestRemoteWorkAPI:
    """Test remote work queries via API."""

    def test_emp004_remote_days(self, api_client):
        """EMP004 should have 3 remote days per week - verify via profile."""
        response = api_client.post(
            "/chat",
            json={
                "message": "How many days can I work remotely per week?",
                "employee_id": "EMP004"
            }
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        profile_call = next((tc for tc in tool_calls if tc.get("tool") == "lookup_employee_profile"), None)
        assert profile_call is not None, "Employee profile should be looked up"
        profile = profile_call.get("result", {}).get("result", {}).get("employee", {})
        assert profile.get("remote_days_per_week") == 3, f"Expected 3 remote days, got {profile}"

    def test_emp001_remote_days(self, api_client):
        """EMP001 should have 2 remote days per week."""
        response = api_client.post(
            "/chat",
            json={
                "message": "What are my remote work options?",
                "employee_id": "EMP001"
            }
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        profile_call = next((tc for tc in tool_calls if tc.get("tool") == "lookup_employee_profile"), None)
        assert profile_call is not None, "Employee profile should be looked up"
        profile = profile_call.get("result", {}).get("result", {}).get("employee", {})
        assert profile.get("remote_days_per_week") == 2, f"Expected 2 remote days, got {profile}"


class TestEmployeeProfileAPI:
    """Test employee profile via API."""

    def test_emp001_profile(self, api_client):
        """EMP001 should be Alice Johnson, Engineering - verify via profile."""
        response = api_client.post(
            "/chat",
            json={
                "message": "Who am I? What is my department?",
                "employee_id": "EMP001"
            }
        )
        assert response.status_code == 200
        data = response.json()
        tool_calls = data.get("tool_calls", [])
        profile_call = next((tc for tc in tool_calls if tc.get("tool") == "lookup_employee_profile"), None)
        assert profile_call is not None, "Employee profile should be looked up"
        profile = profile_call.get("result", {}).get("result", {}).get("employee", {})
        assert profile.get("name") == "Alice Johnson", f"Expected Alice Johnson, got {profile}"
        assert profile.get("department") == "Engineering", f"Expected Engineering, got {profile}"


class TestPolicyQueries:
    """Test policy-only queries."""

    def test_pto_policy_general(self, api_client):
        """General PTO policy should return policy info."""
        response = api_client.post(
            "/chat",
            json={
                "message": "What is the company's PTO policy?"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 50


class TestHealthCheck:
    """Test health endpoint."""

    def test_health(self, api_client):
        """Health check should return OK."""
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
