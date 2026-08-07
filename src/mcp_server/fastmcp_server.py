"""FastMCP server implementation for HR Policy Assistant.

This module provides a standards-compliant MCP server using the FastMCP framework.
FastMCP is the official recommended framework for building MCP servers.
"""

import os
from typing import Optional
from pathlib import Path

from fastmcp import FastMCP

# Import HR tools
from .tools import HRTools, MockDataStore

# Global instances
_mcp: Optional[FastMCP] = None
_hr_tools: Optional[HRTools] = None
_rag_pipeline = None


def set_rag_pipeline(rag_pipeline):
    """Set the RAG pipeline for policy search tools."""
    global _rag_pipeline
    _rag_pipeline = rag_pipeline


def get_rag_pipeline():
    """Get the RAG pipeline."""
    return _rag_pipeline


def get_hr_tools() -> HRTools:
    """Get the HR tools instance."""
    global _hr_tools
    if _hr_tools is None:
        data_dir = os.getenv("MOCK_DATA_DIR", "mock_data")
        _hr_tools = HRTools(data_dir=data_dir)
    return _hr_tools


def create_mcp_server() -> FastMCP:
    """Create and configure the FastMCP server."""
    global _mcp

    _mcp = FastMCP(
        name="HR Policy Assistant",
        instructions="MCP server for HR policy assistance with RAG integration. Provides tools for employee lookup, PTO balance, benefits status, HR tickets, policy compliance checks, and policy document search via RAG.",
    )

    hr = get_hr_tools()

    # Add custom HTTP endpoints for HTTP transport
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @_mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "healthy",
            "server": "HR Policy Assistant (FastMCP)",
            "version": "1.0.0",
            "rag_available": _rag_pipeline is not None
        })

    @_mcp.custom_route("/tools", methods=["GET"])
    async def list_tools_endpoint(request: Request) -> JSONResponse:
        """List all available tools (simple HTTP)."""
        tools = await _mcp.list_tools()
        tool_list = []
        for tool in tools:
            tool_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters
            })
        return JSONResponse({"tools": tool_list})

    @_mcp.custom_route("/tools/call", methods=["POST"])
    async def call_tool_endpoint(request: Request) -> JSONResponse:
        """Call a tool (simple HTTP)."""
        body = await request.json()
        name = body.get("name")
        arguments = body.get("arguments", {})

        try:
            result = await _mcp.call_tool(name, arguments)
            # Convert result to dict
            if hasattr(result, 'content') and result.content:
                import json as json_lib
                for content in result.content:
                    if hasattr(content, 'text'):
                        try:
                            parsed = json_lib.loads(content.text)
                            return JSONResponse({"success": True, "result": parsed})
                        except json_lib.JSONDecodeError:
                            return JSONResponse({"success": True, "text": content.text})
                return JSONResponse({"success": True, "result": str(result.content[0])})
            return JSONResponse({"success": True, "result": None})
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)})

    @_mcp.custom_route("/mcp-api", methods=["POST"])
    async def mcp_api_endpoint(request: Request) -> JSONResponse:
        """Simple HTTP MCP API endpoint (compatible with old clients)."""
        body = await request.json()
        method = body.get("method")
        params = body.get("params", {})

        if method == "tools/list":
            tools = await _mcp.list_tools()
            tool_list = []
            for tool in tools:
                tool_list.append({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.parameters
                })
            return JSONResponse({"result": {"tools": tool_list}})

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            try:
                result = await _mcp.call_tool(tool_name, arguments)
                if hasattr(result, 'content') and result.content:
                    import json as json_lib
                    for content in result.content:
                        if hasattr(content, 'text'):
                            try:
                                parsed = json_lib.loads(content.text)
                                return JSONResponse({"result": parsed})
                            except json_lib.JSONDecodeError:
                                return JSONResponse({"result": {"text": content.text}})
                    return JSONResponse({"result": {"value": str(result.content[0])}})
                return JSONResponse({"result": None})
            except Exception as e:
                return JSONResponse({"error": {"code": -32603, "message": str(e)}})

        return JSONResponse({"error": {"code": -32601, "message": f"Unknown method: {method}"}})

    # =================================================================
    # Employee Profile Tools
    # =================================================================

    @_mcp.tool()
    def lookup_employee_profile(
        employee_id: Optional[str] = None,
        email: Optional[str] = None,
    ) -> dict:
        """Look up an employee's profile information by employee ID or email address.

        Args:
            employee_id: Employee ID (e.g., 'EMP001')
            email: Employee email address

        Returns:
            Employee profile information including department, title, work arrangement, etc.
        """
        return hr.lookup_employee_profile(employee_id=employee_id, email=email)

    # =================================================================
    # PTO Tools
    # =================================================================

    @_mcp.tool()
    def check_pto_balance(employee_id: str, year: int = 2026) -> dict:
        """Check an employee's PTO balance including accrued, used, pending, and available days.

        Args:
            employee_id: Employee ID (e.g., 'EMP001')
            year: Year to check (defaults to 2026)

        Returns:
            PTO balance information with breakdown of accrued, used, pending, and available days.
        """
        return hr.check_pto_balance(employee_id=employee_id, year=year)

    # =================================================================
    # Benefits Tools
    # =================================================================

    @_mcp.tool()
    def lookup_benefits_status(employee_id: str) -> dict:
        """Look up an employee's benefits enrollment status.

        Args:
            employee_id: Employee ID (e.g., 'EMP001')

        Returns:
            Benefits enrollment information including medical, dental, vision, HSA, and FSA.
        """
        return hr.lookup_benefits_status(employee_id=employee_id)

    # =================================================================
    # Ticket and Email Tools
    # =================================================================

    @_mcp.tool()
    def create_mock_hr_ticket(
        employee_id: str,
        category: str,
        subject: str,
        description: str,
    ) -> dict:
        """Create a mock HR ticket for tracking purposes.

        Args:
            employee_id: Employee ID
            category: Ticket category (remote_work, benefits, pto, leave, expense, workplace)
            subject: Ticket subject
            description: Detailed description of the request

        Returns:
            Created ticket information with ticket ID and status.
        """
        return hr.create_mock_hr_ticket(
            employee_id=employee_id,
            category=category,
            subject=subject,
            description=description,
        )

    @_mcp.tool()
    def draft_hr_email(
        employee_id: str,
        purpose: str,
        context: Optional[str] = None,
    ) -> dict:
        """Draft a mock HR email for an employee.

        Args:
            employee_id: Employee ID
            purpose: Email purpose (pto_request, remote_work_approval, benefits_info, general)
            context: Additional context for the email

        Returns:
            Draft email with subject, body, and recipient information.
        """
        return hr.draft_hr_email(
            employee_id=employee_id,
            purpose=purpose,
            context=context,
        )

    # =================================================================
    # Policy Compliance Tools
    # =================================================================

    @_mcp.tool()
    def check_policy_compliance(employee_id: str, policy_area: str) -> dict:
        """Check if an employee is compliant with a specific policy area.

        Args:
            employee_id: Employee ID
            policy_area: Policy area (remote_work, security, equipment)

        Returns:
            Compliance status with requirements checklist.
        """
        return hr.check_policy_compliance(
            employee_id=employee_id,
            policy_area=policy_area,
        )

    # =================================================================
    # RAG-Powered Policy Tools
    # =================================================================

    @_mcp.tool()
    def search_policy_documents(query: str, k: int = 5) -> dict:
        """Search the policy document index for relevant information using RAG.

        Args:
            query: Search query for policy documents
            k: Number of results to return (default 5)

        Returns:
            List of relevant policy document chunks with scores.
        """
        rag = get_rag_pipeline()
        if rag is None or rag.retriever is None:
            return {
                "success": False,
                "error": "RAG pipeline not initialized",
                "query": query,
                "results": [],
            }

        try:
            results = rag.retriever.retrieve(query, k=k)

            formatted_results = []
            for i, chunk in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "document_id": chunk.get("document_id"),
                    "title": chunk.get("title"),
                    "heading": chunk.get("heading", ""),
                    "content": chunk.get("content", ""),
                    "score": chunk.get("score", 0),
                    "source": (chunk.get("metadata") or {}).get("source", ""),
                })

            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "note": "Results from RAG-powered policy search",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": [],
            }

    @_mcp.tool()
    def get_policy_section(policy_title: str, section: Optional[str] = None) -> dict:
        """Get a specific section from a policy document by title using RAG.

        Args:
            policy_title: Title or partial title of the policy document
            section: Optional section name to retrieve

        Returns:
            Policy section content with document metadata.
        """
        rag = get_rag_pipeline()
        if rag is None or rag.retriever is None:
            return {
                "success": False,
                "error": "RAG pipeline not initialized",
                "policy_title": policy_title,
                "section": section,
            }

        try:
            results = rag.retriever.retrieve(policy_title, k=3)

            if not results:
                return {
                    "success": False,
                    "error": f"No policy found matching: {policy_title}",
                    "policy_title": policy_title,
                    "section": section,
                }

            best_match = None
            for chunk in results:
                title = chunk.get("title", "").lower()
                if policy_title.lower() in title or policy_title.lower() in chunk.get("content", "").lower():
                    best_match = chunk
                    break

            if not best_match and results:
                best_match = results[0]

            if best_match:
                content = best_match.get("content", "")
                if section:
                    lines = content.split("\n")
                    section_content = []
                    in_section = False
                    for line in lines:
                        if section.lower() in line.lower():
                            in_section = True
                        if in_section:
                            section_content.append(line)
                            if line.startswith("#") and section.lower() not in line.lower():
                                break

                    if section_content:
                        content = "\n".join(section_content)

                return {
                    "success": True,
                    "policy_title": best_match.get("title"),
                    "document_id": best_match.get("document_id"),
                    "section": section,
                    "content": content,
                    "heading": best_match.get("heading", ""),
                    "note": "Retrieved from RAG-powered policy index",
                }

            return {
                "success": False,
                "error": f"No policy found matching: {policy_title}",
                "policy_title": policy_title,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "policy_title": policy_title,
            }

    return _mcp


def get_mcp_server() -> Optional[FastMCP]:
    """Get the FastMCP server instance."""
    global _mcp
    return _mcp


def run_server(transport: str = "streamable-http", port: int = 8001):
    """Run the MCP server.

    Args:
        transport: Transport type ('streamable-http', 'stdio')
        port: Port for HTTP transport
    """
    mcp = create_mcp_server()

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http", port=port, host="127.0.0.1")
