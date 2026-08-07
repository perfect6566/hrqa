"""Run evaluation on the HR Policy Assistant."""

import json
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set deterministic seed for reproducible evaluation
from evaluation.evaluator import set_evaluation_seed, EVALUATION_SEED
set_evaluation_seed(EVALUATION_SEED)

from src.rag import RAGPipeline
from src.agent import AgentOrchestrator
from evaluation.evaluator import Evaluator, EvaluationResult
from evaluation.questions import load_evaluation_questions


class EvaluationRunner:
    """Run evaluation on the HR Policy Assistant."""

    def __init__(
        self,
        policies_dir: str = "policies",
        mock_data_dir: str = "mock_data",
        mcp_server_url: str = "http://localhost:8001"
    ):
        self.policies_dir = policies_dir
        self.mock_data_dir = mock_data_dir
        self.mcp_server_url = mcp_server_url

        self.rag_pipeline: RAGPipeline = None
        self.orchestrator: AgentOrchestrator = None
        self.evaluator: Evaluator = None

    async def setup(self):
        """Set up the evaluation environment."""
        print("Setting up evaluation environment...")

        # Initialize RAG
        self.rag_pipeline = RAGPipeline(
            policies_dir=self.policies_dir,
            vector_store_path="./data/vector_store"
        )

        # Index documents
        result = self.rag_pipeline.index_documents()
        print(f"Indexing result: {result}")

        # Initialize orchestrator
        self.orchestrator = AgentOrchestrator(
            rag_pipeline=self.rag_pipeline,
            mcp_server_url=self.mcp_server_url
        )

        # Initialize evaluator
        self.evaluator = Evaluator()

        print("Setup complete!")

    async def evaluate_single(self, question_data: Dict) -> EvaluationResult:
        """Evaluate a single question."""
        question_id = question_data["id"]
        question = question_data["question"]
        gold_answer = question_data["gold_answer"]
        employee_id = question_data.get("employee_id")

        print(f"\nEvaluating: {question_id} - {question[:50]}...")

        start_time = asyncio.get_event_loop().time()

        try:
            # Process the question
            result = await self.orchestrator.process_request(
                query=question,
                employee_id=employee_id
            )

            actual_answer = result["answer"]
            citations = result.get("citations", [])
            tool_calls = result.get("tool_calls", [])

        except Exception as e:
            print(f"Error processing question: {e}")
            actual_answer = ""
            citations = []
            tool_calls = []

        end_time = asyncio.get_event_loop().time()
        latency_ms = int((end_time - start_time) * 1000)

        # Evaluate groundedness
        expected_keywords = gold_answer.lower().split()
        groundedness = self._calculate_groundedness(actual_answer, expected_keywords)

        # Evaluate citation accuracy
        citation_accuracy = 1.0 if citations else 0.0

        # Evaluate tool selection
        expected_tool = question_data.get("expected_tool")
        if expected_tool:
            tool_selection_correct = any(
                expected_tool in tc.get("tool", "")
                for tc in tool_calls
            )
        else:
            tool_selection_correct = True

        # Evaluate workflow completion
        workflow = question_data.get("workflow")
        if workflow:
            workflow_completed = len(tool_calls) > 0
        else:
            workflow_completed = True

        return EvaluationResult(
            question_id=question_id,
            question=question,
            expected_answer=gold_answer,
            actual_answer=actual_answer,
            groundedness_score=groundedness,
            citation_accuracy=citation_accuracy,
            tool_selection_correct=tool_selection_correct,
            workflow_completed=workflow_completed,
            latency_ms=latency_ms
        )

    def _calculate_groundedness(self, answer: str, expected_keywords: List[str]) -> float:
        """Calculate groundedness score."""
        answer_lower = answer.lower()
        found = sum(1 for kw in expected_keywords if kw in answer_lower)
        return found / len(expected_keywords) if expected_keywords else 0.0

    async def run_evaluation(self, output_file: str = "evaluation/results.json"):
        """Run full evaluation."""
        questions = load_evaluation_questions()

        print(f"\nRunning evaluation on {len(questions)} questions...")

        for q in questions:
            result = await self.evaluate_single(q)
            self.evaluator.record_result(result)

        # Compute metrics
        metrics = self.evaluator.compute_metrics()

        # Generate report
        report = self.evaluator.generate_report()

        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Total Questions: {report['summary']['total_questions']}")
        print(f"Pass Rate: {report['summary']['pass_rate']}")
        print(f"Groundedness: {report['answer_quality']['groundedness_avg']}")
        print(f"Citation Accuracy: {report['answer_quality']['citation_accuracy_avg']}")
        print(f"Tool Selection: {report['agent_behavior']['tool_selection_accuracy']}")
        print(f"Workflow Completion: {report['agent_behavior']['workflow_completion_rate']}")
        print(f"Latency P50: {report['performance']['latency_p50_ms']}ms")
        print(f"Latency P95: {report['performance']['latency_p95_ms']}ms")

        return report


async def main():
    """Main entry point."""
    runner = EvaluationRunner()
    await runner.setup()
    await runner.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
