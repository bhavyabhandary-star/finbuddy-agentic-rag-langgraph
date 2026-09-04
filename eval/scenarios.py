"""Fixed evaluation scenarios: expected route, expected tools, expected outcome.

Per kickoff_prompt.md's Evaluation Harness: score task completion, correct tool
selection, and step count — not just final-answer text. Split the blame.

TODO: expand once real RBI/DPDP PDFs are ingested (milestone step 2) — the
off-corpus and credit-assessment scenarios are runnable now; the policy scenarios
need real corpus content to assert against specific answer facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    name: str
    query: str
    expected_route: str
    expected_tools: list[str] = field(default_factory=list)
    should_escalate: bool | None = None


SCENARIOS: list[Scenario] = [
    Scenario(
        name="off_topic_escalates_without_llm_call",
        query="what's the weather like today",
        expected_route="off_topic",
        expected_tools=[],
        should_escalate=True,
    ),
    Scenario(
        name="credit_request_routes_to_assessment_tools",
        query="can you check my credit score",
        expected_route="credit_assessment",
        expected_tools=["assess_credit_profile"],
        should_escalate=False,
    ),
    Scenario(
        name="risk_trend_question_routes_to_assessment_tools",
        query="is my financial trend improving or getting worse",
        expected_route="credit_assessment",
        expected_tools=["assess_credit_profile", "assess_risk_trend"],
        should_escalate=False,
    ),
    Scenario(
        name="policy_question_routes_to_rag",
        query="why do you need my UPI transaction data",
        expected_route="policy",
        expected_tools=["retrieve_pdf_chunks"],
        should_escalate=False,
    ),
]
