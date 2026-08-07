"""HR Tools implementation for MCP server."""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime


class MockDataStore:
    """Store for mock HR data."""

    def __init__(self, data_dir: str = "mock_data"):
        self.data_dir = Path(data_dir)
        self._employees: Optional[List[Dict]] = None
        self._pto_balances: Optional[List[Dict]] = None
        self._benefits: Optional[List[Dict]] = None
        self._tickets: Optional[List[Dict]] = None

    def load_employees(self) -> List[Dict]:
        """Load employee data."""
        if self._employees is None:
            with open(self.data_dir / "employees.json", "r") as f:
                self._employees = json.load(f)["employees"]
        return self._employees

    def load_pto_balances(self) -> List[Dict]:
        """Load PTO balance data."""
        if self._pto_balances is None:
            with open(self.data_dir / "pto_balances.json", "r") as f:
                data = json.load(f)
                self._pto_balances = data.get("pto_balances", [])
                self._pto_requests = data.get("pto_requests", [])
        return self._pto_balances

    def load_benefits(self) -> List[Dict]:
        """Load benefits data."""
        if self._benefits is None:
            with open(self.data_dir / "benefits.json", "r") as f:
                self._benefits = json.load(f)["benefits_enrollments"]
        return self._benefits

    def load_tickets(self) -> List[Dict]:
        """Load HR tickets data."""
        if self._tickets is None:
            with open(self.data_dir / "hr_tickets.json", "r") as f:
                self._tickets = json.load(f)["hr_tickets"]
        return self._tickets

    def get_employee_by_id(self, employee_id: str) -> Optional[Dict]:
        """Get employee by ID."""
        employees = self.load_employees()
        for emp in employees:
            if emp["employee_id"] == employee_id:
                return emp
        return None

    def get_employee_by_email(self, email: str) -> Optional[Dict]:
        """Get employee by email."""
        employees = self.load_employees()
        for emp in employees:
            if emp["email"].lower() == email.lower():
                return emp
        return None

    def get_pto_balance(self, employee_id: str, year: int = 2026) -> Optional[Dict]:
        """Get PTO balance for employee."""
        balances = self.load_pto_balances()
        for bal in balances:
            if bal["employee_id"] == employee_id and bal["year"] == year:
                return bal
        return None

    def get_benefits(self, employee_id: str) -> Optional[Dict]:
        """Get benefits enrollment for employee."""
        benefits = self.load_benefits()
        for b in benefits:
            if b["employee_id"] == employee_id:
                return b
        return None


class HRTools:
    """Collection of HR tools for MCP integration."""

    def __init__(self, data_dir: str = "mock_data"):
        self.data_store = MockDataStore(data_dir)

    def lookup_employee_profile(
        self,
        employee_id: Optional[str] = None,
        email: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Look up an employee's profile information.

        Args:
            employee_id: Employee ID (e.g., "EMP001")
            email: Employee email address

        Returns:
            Employee profile information
        """
        if not employee_id and not email:
            return {
                "success": False,
                "error": "Either employee_id or email must be provided"
            }

        employee = None
        if employee_id:
            employee = self.data_store.get_employee_by_id(employee_id)
        elif email:
            employee = self.data_store.get_employee_by_email(email)

        if not employee:
            return {
                "success": False,
                "error": f"Employee not found: {employee_id or email}"
            }

        return {
            "success": True,
            "employee": {
                "employee_id": employee["employee_id"],
                "name": employee["name"],
                "email": employee["email"],
                "department": employee["department"],
                "title": employee["title"],
                "manager_id": employee["manager_id"],
                "hire_date": employee["hire_date"],
                "employment_type": employee["employment_type"],
                "office_location": employee["office_location"],
                "work_arrangement": employee["work_arrangement"],
                "remote_days_per_week": employee.get("remote_days_per_week"),
                "employment_status": employee["employment_status"]
            }
        }

    def check_pto_balance(
        self,
        employee_id: str,
        year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check an employee's PTO balance.

        Args:
            employee_id: Employee ID
            year: Year to check (defaults to 2026)

        Returns:
            PTO balance information including annual entitlement, tenure bracket,
            and progression notes so the assistant can explain why the current
            accrued amount differs from the annual entitlement.
        """
        if year is None:
            year = 2026

        # First verify employee exists
        employee = self.data_store.get_employee_by_id(employee_id)
        if not employee:
            return {
                "success": False,
                "error": f"Employee not found: {employee_id}"
            }

        balance = self.data_store.get_pto_balance(employee_id, year)
        if not balance:
            return {
                "success": False,
                "error": f"No PTO balance found for {employee_id} in {year}"
            }

        # Derive expected accrual based on hire date and elapsed year fraction.
        # PTO policy: 0-2 yr = 15, 2-5 yr = 20, 5+ yr = 25 days/year.
        annual_entitlement = balance.get("annual_entitlement")
        tenure_bracket = balance.get("tenure_bracket")
        if annual_entitlement is None or tenure_bracket is None:
            annual_entitlement, tenure_bracket = self._derive_entitlement(
                employee.get("hire_date"), as_of_year=year
            )

        as_of = datetime(year, 12, 31)
        first_day = datetime(year, 1, 1)
        today = datetime.now()
        if today.year == year:
            as_of = today
        days_elapsed = (as_of - first_day).days + 1
        days_in_year = (datetime(year, 12, 31) - first_day).days + 1
        expected_accrued = round(annual_entitlement * days_elapsed / days_in_year)
        accrued_pct = round(days_elapsed / days_in_year * 100, 1)

        return {
            "success": True,
            "employee_id": employee_id,
            "employee_name": employee["name"],
            "hire_date": employee.get("hire_date"),
            "year": balance["year"],
            "annual_entitlement": annual_entitlement,
            "tenure_bracket": tenure_bracket,
            "accrued_days": balance["accrued_days"],
            "expected_accrued_to_date": expected_accrued,
            "year_progress_pct": accrued_pct,
            "used_days": balance["used_days"],
            "pending_days": balance["pending_days"],
            "available_days": balance["available_days"],
            "carryover": {
                "from_previous": balance["carryover_from_previous"],
                "used": balance["carryover_used"],
                "remaining": balance["carryover_remaining"]
            },
            "explanation_hint": (
                f"Annual entitlement is {annual_entitlement} days per policy "
                f"({tenure_bracket}). As of {as_of.strftime('%Y-%m-%d')} the "
                f"employee is {accrued_pct}% through the year, so expected "
                f"pro-rated accrued ≈ {expected_accrued} days. Current accrued "
                f"= {balance['accrued_days']} days."
            )
        }

    @staticmethod
    def _derive_entitlement(hire_date: Optional[str], as_of_year: int) -> tuple:
        """Derive PTO annual entitlement and tenure bracket from hire_date.

        Returns (annual_entitlement, tenure_bracket). Falls back to (15, 'unknown')
        if hire_date is missing or unparseable.
        """
        if not hire_date:
            return 15, "unknown"
        try:
            hire = datetime.strptime(hire_date, "%Y-%m-%d")
        except (TypeError, ValueError):
            return 15, "unknown"
        years_of_service = as_of_year - hire.year - (
            1 if (as_of_year, 1, 1) < (hire.year, hire.month, hire.day) else 0
        )
        if years_of_service < 2:
            return 15, "0-2 years (15 days/yr)"
        if years_of_service < 5:
            return 20, "2-5 years (20 days/yr)"
        return 25, "5+ years (25 days/yr)"

    def lookup_benefits_status(
        self,
        employee_id: str
    ) -> Dict[str, Any]:
        """
        Look up an employee's benefits enrollment status.

        Args:
            employee_id: Employee ID

        Returns:
            Benefits enrollment information
        """
        employee = self.data_store.get_employee_by_id(employee_id)
        if not employee:
            return {
                "success": False,
                "error": f"Employee not found: {employee_id}"
            }

        benefits = self.data_store.get_benefits(employee_id)
        if not benefits:
            return {
                "success": False,
                "error": f"No benefits enrollment found for {employee_id}"
            }

        return {
            "success": True,
            "employee_id": employee_id,
            "employee_name": employee["name"],
            "enrollment_date": benefits["enrollment_date"],
            "medical": {
                "plan": benefits["medical_plan"],
                "tier": benefits["medical_tier"]
            },
            "dental_enrolled": benefits["dental_enrolled"],
            "vision_enrolled": benefits["vision_enrolled"],
            "hsa": {
                "enrolled": benefits.get("hsa_enrolled", False),
                "company_contribution": benefits.get("hsa_company_contribution", 0),
                "employee_contribution": benefits.get("hsa_employee_contribution", 0)
            } if benefits.get("hsa_enrolled") else {"enrolled": False},
            "fsa": {
                "healthcare": {
                    "enrolled": benefits.get("fsa_healthcare_enrolled", False),
                    "contribution": benefits.get("fsa_healthcare_contribution", 0)
                },
                "dependent_care": {
                    "enrolled": benefits.get("fsa_dependent_care_enrolled", False),
                    "contribution": benefits.get("fsa_dependent_care_contribution", 0)
                }
            },
            "annual_cost": benefits["annual_benefits_cost"]
        }

    def create_mock_hr_ticket(
        self,
        employee_id: str,
        category: str,
        subject: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Create a mock HR ticket (simulated, not actually created in a system).

        Args:
            employee_id: Employee ID
            category: Ticket category (remote_work, benefits, pto, leave, expense, workplace)
            subject: Ticket subject
            description: Ticket description

        Returns:
            Created ticket information
        """
        employee = self.data_store.get_employee_by_id(employee_id)
        if not employee:
            return {
                "success": False,
                "error": f"Employee not found: {employee_id}"
            }

        valid_categories = ["remote_work", "benefits", "pto", "leave", "expense", "workplace"]
        if category not in valid_categories:
            return {
                "success": False,
                "error": f"Invalid category. Must be one of: {', '.join(valid_categories)}"
            }

        # Generate mock ticket
        ticket_id = f"HR{len(self.data_store.load_tickets()) + 1:03d}"
        now = datetime.now().strftime("%Y-%m-%d")

        return {
            "success": True,
            "message": "Mock HR ticket created successfully (this is a simulation)",
            "ticket": {
                "ticket_id": ticket_id,
                "employee_id": employee_id,
                "employee_name": employee["name"],
                "category": category,
                "subject": subject,
                "description": description,
                "status": "pending",
                "created_date": now,
                "assigned_to": "EMP003",
                "note": "This is a mock ticket for demonstration purposes. In production, this would be created in the HR system."
            }
        }

    def draft_hr_email(
        self,
        employee_id: str,
        purpose: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Draft a mock HR email for an employee.

        Args:
            employee_id: Employee ID
            purpose: Purpose of the email (pto_request, remote_work_approval, benefits_info, general)
            context: Additional context for the email

        Returns:
            Draft email content
        """
        employee = self.data_store.get_employee_by_id(employee_id)
        if not employee:
            return {
                "success": False,
                "error": f"Employee not found: {employee_id}"
            }

        templates = {
            "pto_request": {
                "subject": f"PTO Request - {employee['name']}",
                "body": f"""Dear {employee['name']},

This email confirms your PTO request has been submitted for manager approval.

Employee: {employee['name']} ({employee['employee_id']})
Department: {employee['department']}
Request Date: {datetime.now().strftime('%Y-%m-%d')}

Your manager will review your request and respond within 2 business days.

Best regards,
HR Department

---
This is a mock email for demonstration purposes."""
            },
            "remote_work_approval": {
                "subject": f"Remote Work Request - {employee['name']}",
                "body": f"""Dear {employee['name']},

Your remote work request has been received and is under review.

Employee: {employee['name']} ({employee['employee_id']})
Current Work Arrangement: {employee['work_arrangement']}
Location: {employee['office_location']}

Our HR team will review your request and any location-related requirements within 5 business days.

Best regards,
HR Department

---
This is a mock email for demonstration purposes."""
            },
            "benefits_info": {
                "subject": f"Benefits Information - {employee['name']}",
                "body": f"""Dear {employee['name']},

Thank you for your inquiry about benefits. Here is a summary of your current enrollment:

Current Enrollment: {employee.get('benefits', {}).get('medical_plan', 'See benefits portal')}

For detailed information about your benefits, please visit the HR benefits portal or contact benefits@company.com.

Best regards,
HR Department

---
This is a mock email for demonstration purposes."""
            },
            "general": {
                "subject": f"HR Inquiry Response - {employee['name']}",
                "body": f"""Dear {employee['name']},

Thank you for reaching out to HR. We have received your inquiry and will respond within 2 business days.

Employee: {employee['name']} ({employee['employee_id']})
Department: {employee['department']}

If you have an urgent matter, please call the HR hotline at 1-800-XXX-XXXX.

Best regards,
HR Department

---
This is a mock email for demonstration purposes."""
            }
        }

        template = templates.get(purpose, templates["general"])

        return {
            "success": True,
            "message": "Mock email drafted successfully",
            "email": {
                "to": employee["email"],
                "subject": template["subject"],
                "body": template["body"],
                "cc": "hr@company.com",
                "note": "This is a mock email for demonstration purposes"
            }
        }

    def check_policy_compliance(
        self,
        employee_id: str,
        policy_area: str
    ) -> Dict[str, Any]:
        """
        Check if an employee is compliant with a specific policy area.

        Args:
            employee_id: Employee ID
            policy_area: Policy area to check (remote_work, security, equipment, etc.)

        Returns:
            Compliance status information
        """
        employee = self.data_store.get_employee_by_id(employee_id)
        if not employee:
            return {
                "success": False,
                "error": f"Employee not found: {employee_id}"
            }

        policy_checks = {
            "remote_work": {
                "compliant": True,
                "requirements": [
                    {"check": "Completed onboarding period", "status": "pass"},
                    {"check": "Eligible work arrangement", "status": "pass"},
                    {"check": "Remote work setup", "status": employee["work_arrangement"] in ["remote", "hybrid"]}
                ],
                "notes": f"Current arrangement: {employee['work_arrangement']}"
            },
            "security": {
                "compliant": True,
                "requirements": [
                    {"check": "MFA enabled", "status": "assumed_pass"},
                    {"check": "VPN access", "status": "assumed_pass"},
                    {"check": "Security training", "status": "assumed_pass"}
                ],
                "notes": "Based on standard onboarding compliance"
            },
            "equipment": {
                "compliant": True,
                "requirements": [
                    {"check": "Company equipment assigned", "status": "pass"},
                    {"check": "Equipment return pending", "status": employee["employment_status"] == "active"}
                ],
                "notes": f"Equipment assigned per standard policy for {employee['employment_type']} employees"
            }
        }

        if policy_area not in policy_checks:
            return {
                "success": False,
                "error": f"Unknown policy area: {policy_area}. Valid areas: {', '.join(policy_checks.keys())}"
            }

        result = policy_checks[policy_area].copy()
        result["employee_id"] = employee_id
        result["employee_name"] = employee["name"]
        result["policy_area"] = policy_area
        result["checked_date"] = datetime.now().strftime("%Y-%m-%d")

        return result


# Global instance
hr_tools = HRTools()


def get_hr_tools() -> HRTools:
    """Get the HR tools instance."""
    return hr_tools
