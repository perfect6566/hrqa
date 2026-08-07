"""Tests for API endpoints."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_api_imports(self):
        """Test that API can be imported."""
        from src.api import app
        assert app is not None

    def test_app_routes_exist(self):
        """Test that required routes are registered."""
        from src.api.main import app

        routes = [r.path for r in app.routes]
        assert "/health" in routes
        assert "/chat" in routes
        assert "/" in routes

    def test_demo_endpoints_exist(self):
        """Test that demo endpoints are registered."""
        from src.api.main import app

        routes = [r.path for r in app.routes]
        assert "/demo/pto-request" in routes
        assert "/demo/remote-work" in routes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
