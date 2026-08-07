"""Evaluation questions and expected answers."""

EVALUATION_QUESTIONS = [
    # Policy Q&A (Simple)
    {
        "id": "eval_01",
        "question": "How much PTO do I earn as a new full-time employee?",
        "gold_answer": "New full-time employees earn 15 days of PTO per year.",
        "category": "policy_qa",
        "difficulty": "easy",
        "requires_tools": False
    },
    {
        "id": "eval_02",
        "question": "What is the maximum PTO I can carry over to the next year?",
        "gold_answer": "Employees may carry over up to 5 unused PTO days to the following year.",
        "category": "policy_qa",
        "difficulty": "easy",
        "requires_tools": False
    },
    {
        "id": "eval_03",
        "question": "What equipment does the company provide for remote workers?",
        "gold_answer": "The company provides laptop, monitor, keyboard, mouse, and headset.",
        "category": "policy_qa",
        "difficulty": "easy",
        "requires_tools": False
    },
    {
        "id": "eval_04",
        "question": "How much is the monthly internet stipend for remote workers?",
        "gold_answer": "Remote workers receive a $50 monthly internet stipend.",
        "category": "policy_qa",
        "difficulty": "easy",
        "requires_tools": False
    },
    {
        "id": "eval_05",
        "question": "What are the data classification levels?",
        "gold_answer": "Data is classified as Public, Internal, Confidential, or Restricted.",
        "category": "policy_qa",
        "difficulty": "medium",
        "requires_tools": False
    },
    # Employee Data (Tool-Required)
    {
        "id": "eval_06",
        "question": "What is Alice Johnson's PTO balance?",
        "gold_answer": "Alice Johnson has 8 available PTO days.",
        "category": "employee_data",
        "difficulty": "easy",
        "requires_tools": True,
        "employee_id": "EMP001",
        "expected_tool": "check_pto_balance"
    },
    {
        "id": "eval_07",
        "question": "What is Bob Smith's benefits status?",
        "gold_answer": "Bob Smith is enrolled in HDHP with HSA.",
        "category": "employee_data",
        "difficulty": "easy",
        "requires_tools": True,
        "employee_id": "EMP002",
        "expected_tool": "lookup_benefits_status"
    },
    {
        "id": "eval_08",
        "question": "Where does Carol Davis work?",
        "gold_answer": "Carol Davis works in San Francisco in a hybrid arrangement.",
        "category": "employee_data",
        "difficulty": "easy",
        "requires_tools": True,
        "employee_id": "EMP003",
        "expected_tool": "lookup_employee_profile"
    },
    {
        "id": "eval_09",
        "question": "What is David Wilson's department?",
        "gold_answer": "David Wilson works in Marketing.",
        "category": "employee_data",
        "difficulty": "easy",
        "requires_tools": True,
        "employee_id": "EMP004",
        "expected_tool": "lookup_employee_profile"
    },
    {
        "id": "eval_10",
        "question": "What is Emma Brown's work arrangement?",
        "gold_answer": "Emma Brown works on-site.",
        "category": "employee_data",
        "difficulty": "easy",
        "requires_tools": True,
        "employee_id": "EMP005",
        "expected_tool": "lookup_employee_profile"
    },
    # Multi-Document
    {
        "id": "eval_11",
        "question": "What are the requirements for working remotely and what equipment is provided?",
        "gold_answer": "Requirements include completing 90-day onboarding, reliable internet, and VPN usage. Equipment includes laptop, monitor, and headset.",
        "category": "multi_doc",
        "difficulty": "hard",
        "requires_tools": False,
        "policy_docs": ["remote-work-policy", "equipment-policy"]
    },
    {
        "id": "eval_12",
        "question": "How do expense reimbursement and remote work policies interact?",
        "gold_answer": "Remote workers can expense home office equipment up to $500 and receive $50/month internet stipend.",
        "category": "multi_doc",
        "difficulty": "hard",
        "requires_tools": False,
        "policy_docs": ["expense-policy", "remote-work-policy"]
    },
    # Agentic Workflows
    {
        "id": "eval_13",
        "question": "Can I take 3 days of PTO next week?",
        "gold_answer": "Check your PTO balance and submit request. For 3+ days, manager approval is required.",
        "category": "workflow",
        "difficulty": "medium",
        "requires_tools": True,
        "workflow": "pto_request",
        "expected_tools": ["check_pto_balance", "search_policy_documents"]
    },
    {
        "id": "eval_14",
        "question": "Am I eligible to work remotely from another state for 6 weeks?",
        "gold_answer": "Full-time remote workers can work from any state with 2-week notice. International requires 4-week approval.",
        "category": "workflow",
        "difficulty": "hard",
        "requires_tools": True,
        "workflow": "remote_work_eligibility",
        "expected_tools": ["lookup_employee_profile", "check_policy_compliance", "search_policy_documents"]
    },
    # Ambiguous Requests
    {
        "id": "eval_15",
        "question": "Tell me about vacation",
        "gold_answer": "Should ask for clarification or provide general PTO information.",
        "category": "ambiguous",
        "difficulty": "medium",
        "expected_behavior": "clarify_or_broad"
    },
    # Out of Scope
    {
        "id": "eval_16",
        "question": "What's the company's stock price?",
        "gold_answer": "Stock price is outside HR scope.",
        "category": "out_of_scope",
        "difficulty": "easy",
        "expected_behavior": "out_of_scope_response"
    },
    {
        "id": "eval_17",
        "question": "Can you give me legal advice?",
        "gold_answer": "Legal advice is outside HR scope.",
        "category": "out_of_scope",
        "difficulty": "easy",
        "expected_behavior": "out_of_scope_response"
    },
    # Complex Employee Workflows
    {
        "id": "eval_18",
        "question": "I want to add my spouse to my medical plan. Can I do that?",
        "gold_answer": "Changes require qualifying life event. Open enrollment is Nov 1-30.",
        "category": "workflow",
        "difficulty": "medium",
        "requires_tools": True,
        "workflow": "benefits_question",
        "expected_tools": ["lookup_benefits_status", "search_policy_documents"]
    },
    {
        "id": "eval_19",
        "question": "What happens if I need to work from a public WiFi location?",
        "gold_answer": "Use VPN when on public WiFi for any work, especially with sensitive data.",
        "category": "policy_qa",
        "difficulty": "medium",
        "requires_tools": False
    },
    {
        "id": "eval_20",
        "question": "How do I submit an expense report?",
        "gold_answer": "Submit through expense portal within 30 days with itemized receipts.",
        "category": "policy_qa",
        "difficulty": "easy",
        "requires_tools": False
    }
]


def load_evaluation_questions() -> list:
    """Load evaluation questions."""
    return EVALUATION_QUESTIONS


def get_questions_by_category(category: str) -> list:
    """Get questions by category."""
    return [q for q in EVALUATION_QUESTIONS if q.get("category") == category]


def get_agentic_questions() -> list:
    """Get questions requiring agentic workflows."""
    return [q for q in EVALUATION_QUESTIONS if q.get("requires_tools")]


def get_workflow_questions() -> list:
    """Get workflow questions."""
    return get_questions_by_category("workflow")
