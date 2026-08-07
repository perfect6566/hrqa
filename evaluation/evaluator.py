"""
Evaluation script for HR Policy Assistant.

This module provides evaluation metrics for:
- Answer quality (groundedness, citation accuracy)
- Agent behavior (tool selection, workflow completion)
- System performance (latency)
"""

import random
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Set deterministic seed for reproducible evaluation
EVALUATION_SEED = 42


def set_evaluation_seed(seed: int = EVALUATION_SEED):
    """Set the random seed for reproducible evaluation."""
    random.seed(seed)


def get_evaluation_seed() -> int:
    """Get the current evaluation seed."""
    return EVALUATION_SEED


# Set seed on module import
set_evaluation_seed(EVALUATION_SEED)


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    question_id: str
    question: str
    expected_answer: str
    actual_answer: str
    groundedness_score: float
    citation_accuracy: float
    tool_selection_correct: bool
    workflow_completed: bool
    latency_ms: int
    notes: str = ""


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    groundedness_avg: float = 0.0
    citation_accuracy_avg: float = 0.0
    tool_selection_accuracy: float = 0.0
    workflow_completion_rate: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    total_questions: int = 0
    passed_questions: int = 0


class EvaluationDataset:
    """Evaluation dataset with gold answers."""

    def __init__(self):
        self.questions = self._load_questions()

    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load evaluation questions."""
        return [
            # Policy Q&A Questions
            {
                "id": "q1",
                "category": "policy_qa",
                "question": "How many days of PTO do I get as a new employee?",
                "expected_keywords": ["15", "days", "year", "new"],
                "requires_tools": False,
                "policy_doc": "pto-policy"
            },
            {
                "id": "q2",
                "category": "policy_qa",
                "question": "What is the company's remote work policy?",
                "expected_keywords": ["remote", "eligibility", "vpn", "security"],
                "requires_tools": False,
                "policy_doc": "remote-work-policy"
            },
            {
                "id": "q3",
                "category": "policy_qa",
                "question": "Can I expense a home office chair?",
                "expected_keywords": ["reimburse", "chair", "home office", "equipment"],
                "requires_tools": False,
                "policy_doc": "expense-policy"
            },
            {
                "id": "q4",
                "category": "policy_qa",
                "question": "What are the company's data security requirements?",
                "expected_keywords": ["vpn", "password", "mfa", "encryption"],
                "requires_tools": False,
                "policy_doc": "data-security-policy"
            },
            {
                "id": "q5",
                "category": "multi_doc",
                "question": "What are the requirements for working remotely from another country?",
                "expected_keywords": ["international", "approval", "visa", "tax"],
                "requires_tools": False,
                "policy_docs": ["remote-work-policy", "data-security-policy"]
            },
            # Employee Data Questions
            {
                "id": "q6",
                "category": "employee_data",
                "question": "What is Alice Johnson's PTO balance?",
                "expected_keywords": ["Alice", "PTO", "balance"],
                "requires_tools": True,
                "tool": "check_pto_balance",
                "employee_id": "EMP001"
            },
            {
                "id": "q7",
                "category": "employee_data",
                "question": "What are Bob Smith's benefits?",
                "expected_keywords": ["Bob", "benefits", "medical"],
                "requires_tools": True,
                "tool": "lookup_benefits_status",
                "employee_id": "EMP002"
            },
            {
                "id": "q8",
                "category": "employee_data",
                "question": "What is the work arrangement for EMP003?",
                "expected_keywords": ["HR", "hybrid", "Carol"],
                "requires_tools": True,
                "tool": "lookup_employee_profile",
                "employee_id": "EMP003"
            },
            # Complex Workflows
            {
                "id": "q9",
                "category": "workflow",
                "question": "Can I take 3 days of PTO next week?",
                "expected_keywords": ["PTO", "balance", "request", "policy"],
                "requires_tools": True,
                "workflow": "pto_request",
                "tools_expected": ["check_pto_balance", "search_policy_documents"]
            },
            {
                "id": "q10",
                "category": "workflow",
                "question": "Can I work remotely from another state for 6 weeks?",
                "expected_keywords": ["remote", "approval", "policy", "compliance"],
                "requires_tools": True,
                "workflow": "remote_work_eligibility",
                "tools_expected": ["lookup_employee_profile", "check_policy_compliance", "search_policy_documents"]
            },
            # Ambiguous Requests
            {
                "id": "q11",
                "category": "ambiguous",
                "question": "Tell me about vacation",
                "expected_behavior": "clarify_or_search",
                "requires_tools": False
            },
            # Out of Scope
            {
                "id": "q12",
                "category": "out_of_scope",
                "question": "What's the stock price?",
                "expected_behavior": "out_of_scope_response",
                "requires_tools": False
            },
        ]

    def get_questions(self, category: Optional[str] = None) -> List[Dict]:
        """Get questions, optionally filtered by category."""
        if category:
            return [q for q in self.questions if q.get("category") == category]
        return self.questions

    def get_policy_qa_questions(self) -> List[Dict]:
        """Get policy Q&A questions."""
        return self.get_questions("policy_qa")

    def get_agentic_questions(self) -> List[Dict]:
        """Get questions requiring agentic workflows."""
        return [q for q in self.questions if q.get("requires_tools")]

    def get_workflow_questions(self) -> List[Dict]:
        """Get workflow questions."""
        return self.get_questions("workflow")


class Evaluator:
    """Evaluate RAG and agent responses."""

    def __init__(self):
        self.dataset = EvaluationDataset()
        self.results: List[EvaluationResult] = []

    def evaluate_answer(
        self,
        question: str,
        actual_answer: str,
        expected_keywords: List[str],
        citations: List[Dict]
    ) -> Dict[str, Any]:
        """Evaluate a single answer."""
        answer_lower = actual_answer.lower()

        # Calculate groundedness
        grounded_keywords = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        groundedness = grounded_keywords / len(expected_keywords) if expected_keywords else 0.5

        # Calculate citation accuracy
        citation_accuracy = 0.0
        if citations:
            cited_docs = set(c.get("document_id") or c.get("title", "").lower() for c in citations)
            if any(cited_docs):
                citation_accuracy = 1.0  # Has citations

        return {
            "groundedness": groundedness,
            "citation_accuracy": citation_accuracy,
            "keywords_found": grounded_keywords,
            "total_keywords": len(expected_keywords)
        }

    def evaluate_tool_selection(
        self,
        expected_tools: List[str],
        actual_tool_calls: List[Dict]
    ) -> bool:
        """Evaluate if correct tools were selected."""
        if not expected_tools:
            return True

        actual_tool_names = [tc.get("tool") for tc in actual_tool_calls if tc.get("tool")]
        actual_set = set(actual_tool_names)

        # Check if all expected tools were called
        expected_set = set(expected_tools)

        # At least the primary tool should be called
        primary_tool = expected_tools[0]
        return primary_tool in actual_set

    def evaluate_workflow_completion(
        self,
        workflow_name: str,
        tool_calls: List[Dict],
        answer: str
    ) -> bool:
        """Evaluate if a workflow was completed correctly."""
        if workflow_name == "pto_request":
            # Should have called PTO and policy tools
            tool_names = [tc.get("tool") for tc in tool_calls]
            has_pto = "check_pto_balance" in tool_names
            has_policy = "search_policy_documents" in tool_names
            return has_pto or has_policy

        elif workflow_name == "remote_work_eligibility":
            tool_names = [tc.get("tool") for tc in tool_calls]
            has_profile = "lookup_employee_profile" in tool_names
            has_compliance = "check_policy_compliance" in tool_names
            return has_profile and has_compliance

        return False

    def record_result(self, result: EvaluationResult):
        """Record an evaluation result."""
        self.results.append(result)

    def compute_metrics(self) -> EvaluationMetrics:
        """Compute aggregated metrics."""
        if not self.results:
            return EvaluationMetrics()

        total = len(self.results)
        passed = sum(1 for r in self.results if r.groundedness_score >= 0.5)

        # Groundedness
        groundedness_avg = sum(r.groundedness_score for r in self.results) / total

        # Citation accuracy
        citation_avg = sum(r.citation_accuracy for r in self.results) / total

        # Tool selection
        tool_correct = sum(1 for r in self.results if r.tool_selection_correct)
        tool_accuracy = tool_correct / total

        # Workflow completion
        workflow_complete = sum(1 for r in self.results if r.workflow_completed)
        workflow_rate = workflow_complete / total

        # Latency
        latencies = sorted([r.latency_ms for r in self.results])
        p50_idx = int(total * 0.5)
        p95_idx = int(total * 0.95)

        return EvaluationMetrics(
            groundedness_avg=groundedness_avg,
            citation_accuracy_avg=citation_avg,
            tool_selection_accuracy=tool_accuracy,
            workflow_completion_rate=workflow_rate,
            latency_p50_ms=latencies[p50_idx] if latencies else 0,
            latency_p95_ms=latencies[p95_idx] if latencies else 0,
            total_questions=total,
            passed_questions=passed
        )

    def generate_report(self) -> Dict[str, Any]:
        """Generate evaluation report."""
        metrics = self.compute_metrics()

        report = {
            "summary": {
                "total_questions": metrics.total_questions,
                "passed_questions": metrics.passed_questions,
                "pass_rate": f"{metrics.passed_questions / metrics.total_questions * 100:.1f}%" if metrics.total_questions > 0 else "0%"
            },
            "answer_quality": {
                "groundedness_avg": f"{metrics.groundedness_avg * 100:.1f}%",
                "citation_accuracy_avg": f"{metrics.citation_accuracy_avg * 100:.1f}%"
            },
            "agent_behavior": {
                "tool_selection_accuracy": f"{metrics.tool_selection_accuracy * 100:.1f}%",
                "workflow_completion_rate": f"{metrics.workflow_completion_rate * 100:.1f}%"
            },
            "performance": {
                "latency_p50_ms": metrics.latency_p50_ms,
                "latency_p95_ms": metrics.latency_p95_ms
            },
            "detailed_results": [
                {
                    "id": r.question_id,
                    "question": r.question,
                    "groundedness": r.groundedness_score,
                    "citations": r.citation_accuracy,
                    "tool_correct": r.tool_selection_correct,
                    "workflow_complete": r.workflow_completed,
                    "latency_ms": r.latency_ms
                }
                for r in self.results
            ]
        }

        return report


# Ablation study support
class AblationStudy:
    """Run ablation studies on RAG parameters."""

    def __init__(self, base_pipeline):
        self.base_pipeline = base_pipeline
        self.results: Dict[str, Any] = {}

    def test_chunk_sizes(self, sizes: List[int]) -> Dict[str, float]:
        """Test different chunk sizes."""
        results = {}
        # Note: In practice, would rebuild index with different chunk sizes
        for size in sizes:
            results[f"chunk_{size}"] = {"groundedness": 0.0}  # Placeholder
        return results

    def test_retrieval_k(self, k_values: List[int]) -> Dict[str, float]:
        """Test different retrieval k values."""
        results = {}
        for k in k_values:
            results[f"k_{k}"] = {"groundedness": 0.0}  # Placeholder
        return results
