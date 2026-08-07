"""Unit tests for HR Policy Assistant."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDocumentLoader:
    """Test document loading functionality."""

    def test_load_documents(self):
        """Test loading policy documents."""
        from src.rag.document_loader import DocumentLoader

        loader = DocumentLoader("policies")
        docs = loader.load_documents()

        assert len(docs) > 0, "No documents loaded"
        assert all("content" in doc for doc in docs)
        assert all("title" in doc for doc in docs)
        assert all("id" in doc for doc in docs)

    def test_document_has_sections(self):
        """Test that documents have sections."""
        from src.rag.document_loader import DocumentLoader

        loader = DocumentLoader("policies")
        docs = loader.load_documents()

        if len(docs) > 0:
            doc = docs[0]
            assert "sections" in doc
            assert isinstance(doc["sections"], list)


class TestDocumentChunker:
    """Test document chunking functionality."""

    def test_chunk_documents(self):
        """Test chunking documents."""
        from src.rag.document_loader import DocumentLoader
        from src.rag.chunker import DocumentChunker

        loader = DocumentLoader("policies")
        docs = loader.load_documents()

        if len(docs) > 0:
            chunker = DocumentChunker(chunk_size=200)
            chunks = chunker.chunk_documents(docs)

            assert len(chunks) > 0, "No chunks created"
            assert all("content" in chunk for chunk in chunks)
            assert all("id" in chunk for chunk in chunks)

    def test_chunk_stats(self):
        """Test chunk statistics."""
        from src.rag.document_loader import DocumentLoader
        from src.rag.chunker import DocumentChunker

        loader = DocumentLoader("policies")
        docs = loader.load_documents()

        if len(docs) > 0:
            chunker = DocumentChunker(chunk_size=200)
            chunks = chunker.chunk_documents(docs)
            stats = chunker.get_stats(chunks)

            assert stats["total_chunks"] > 0
            assert stats["avg_length_chars"] > 0


class TestHRTools:
    """Test HR MCP tools."""

    def test_lookup_employee_profile(self):
        """Test employee profile lookup."""
        from src.mcp.tools import HRTools

        tools = HRTools(data_dir="mock_data")

        result = tools.lookup_employee_profile(employee_id="EMP001")

        assert result["success"] is True
        assert "employee" in result
        assert result["employee"]["name"] == "Alice Johnson"
        assert result["employee"]["department"] == "Engineering"

    def test_lookup_employee_by_email(self):
        """Test employee lookup by email."""
        from src.mcp.tools import HRTools

        tools = HRTools(data_dir="mock_data")

        result = tools.lookup_employee_profile(email="alice.johnson@company.com")

        assert result["success"] is True
        assert result["employee"]["employee_id"] == "EMP001"

    def test_employee_not_found(self):
        """Test handling of non-existent employee."""
        from src.mcp.tools import HRTools

        tools = HRTools(data_dir="mock_data")

        result = tools.lookup_employee_profile(employee_id="INVALID")

        assert result["success"] is False
        assert "error" in result

    def test_check_pto_balance(self):
        """Test PTO balance check."""
        from src.mcp.tools import HRTools

        tools = HRTools(data_dir="mock_data")

        result = tools.check_pto_balance(employee_id="EMP001")

        assert result["success"] is True
        assert result["employee_name"] == "Alice Johnson"
        assert "available_days" in result
        assert "accrued_days" in result

    def test_lookup_benefits_status(self):
        """Test benefits status lookup."""
        from src.mcp.tools import HRTools

        tools = HRTools(data_dir="mock_data")

        result = tools.lookup_benefits_status(employee_id="EMP001")

        assert result["success"] is True
        assert result["employee_name"] == "Alice Johnson"
        assert "medical" in result

    def test_create_mock_hr_ticket(self):
        """Test HR ticket creation."""
        from src.mcp.tools import HRTools

        tools = HRTools(data_dir="mock_data")

        result = tools.create_mock_hr_ticket(
            employee_id="EMP001",
            category="pto",
            subject="Test Request",
            description="Testing ticket creation"
        )

        assert result["success"] is True
        assert "ticket" in result
        assert result["ticket"]["employee_id"] == "EMP001"
        assert result["ticket"]["status"] == "pending"

    def test_draft_hr_email(self):
        """Test HR email drafting."""
        from src.mcp.tools import HRTools

        tools = HRTools(data_dir="mock_data")

        result = tools.draft_hr_email(
            employee_id="EMP001",
            purpose="pto_request"
        )

        assert result["success"] is True
        assert "email" in result
        assert "to" in result["email"]
        assert "subject" in result["email"]
        assert "body" in result["email"]

    def test_check_policy_compliance(self):
        """Test policy compliance check."""
        from src.mcp.tools import HRTools

        tools = HRTools(data_dir="mock_data")

        result = tools.check_policy_compliance(
            employee_id="EMP001",
            policy_area="remote_work"
        )

        # check_policy_compliance returns compliance status directly
        assert "compliant" in result
        assert result["employee_id"] == "EMP001"
        assert "requirements" in result


class TestEmbedder:
    """Test embedding functionality."""

    def test_embedder_initialization(self):
        """Test embedder can be initialized."""
        import os
        from src.rag.embedder import Embedder

        if os.getenv("OPENAI_API_KEY"):
            embedder = Embedder()
            assert embedder.model is not None
            assert embedder.get_embedding_dimension() > 0


class TestMCPServer:
    """Test MCP server."""

    def test_mcp_server_creation(self):
        """Test MCP server can be created."""
        from src.mcp.server import MCPServer

        server = MCPServer(name="test-server")
        assert server.name == "test-server"

    def test_tool_decorator(self):
        """Test tool registration decorator."""
        from src.mcp.server import MCPServer

        server = MCPServer(name="test-server")

        @server.tool(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"}
        )
        def test_handler():
            return {"result": "success"}

        tools = server.list_tools()
        assert any(t["name"] == "test_tool" for t in tools)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
