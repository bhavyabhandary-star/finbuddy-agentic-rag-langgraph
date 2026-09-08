"""Smoke tests: the graph compiles and off-topic queries never reach the LLM.

Full end-to-end tests (real retrieval, real Claude calls) belong in the eval
harness (eval/evaluate_agent.py), not here — these are fast, no-network unit tests.
"""
from unittest.mock import patch

from agent.graph import build_graph
from tools.schemas import CreditAssessmentResult


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_off_topic_query_escalates_without_llm_call():
    graph = build_graph()
    with patch("agent.nodes.route.classify_intent", return_value=("off_topic", 0.9)):
        with patch("agent.graph.generate_node") as mock_generate:
            result = graph.invoke({"query": "what's the weather today", "session_id": "test-session"})

    mock_generate.assert_not_called()
    assert result["escalate_to_human"] is True


def test_credit_assessment_tool_failure_escalates_without_llm_call():
    """Regression test for a real bug found during a guardrail review (Session
    18's "wrong information" / decision-support-not-autonomous-decision-maker
    theme): a failed assess_credit_profile call used to still flow to Generate,
    which would narrate the tool's fallback zero-data as if it were real.
    """
    graph = build_graph()
    failed_result = CreditAssessmentResult(
        credit_score=0,
        calibrated_probability_of_repayment=0.0,
        approved=False,
        fairness_mitigation_applied=False,
        income_band="unknown",
        is_anomalous=False,
        anomaly_note=None,
        top_3_factors=[],
        latency_ms=0.0,
        tool_error="ConnectionError: down",
    )

    with patch("agent.nodes.route.classify_intent", return_value=("credit_assessment", 0.9)):
        with patch("agent.graph.credit_assessment_node") as mock_credit_node:
            mock_credit_node.side_effect = lambda s, **kw: {**s, "credit_assessment": failed_result, "risk_trend": None}
            with patch("agent.graph.generate_node") as mock_generate:
                result = graph.invoke({"query": "can you check my credit score", "session_id": "test-session"})

    mock_generate.assert_not_called()
    assert result["escalate_to_human"] is True
    assert "0" not in result["answer"]  # never narrates the fallback zero-score as real
    assert "scoring system" in result["answer"].lower()


def test_credit_assessment_success_gets_disclaimer():
    graph = build_graph()
    ok_result = CreditAssessmentResult(
        credit_score=738,
        calibrated_probability_of_repayment=0.81,
        approved=True,
        fairness_mitigation_applied=True,
        income_band="middle",
        is_anomalous=False,
        anomaly_note=None,
        top_3_factors=[],
        latency_ms=85.0,
    )

    with patch("agent.nodes.route.classify_intent", return_value=("credit_assessment", 0.9)):
        with patch("agent.graph.credit_assessment_node") as mock_credit_node:
            mock_credit_node.side_effect = lambda s, **kw: {**s, "credit_assessment": ok_result, "risk_trend": None}
            with patch("agent.graph.generate_node") as mock_generate:
                mock_generate.side_effect = lambda s, **kw: {
                    **s,
                    "answer": "your score is 738",
                    "sources": [],
                    "confidence": 0.9,
                    "escalate_to_human": False,
                    "disclaimer": "This is an AI-generated explanation of an automated assessment, not financial advice.",
                }
                result = graph.invoke({"query": "can you check my credit score", "session_id": "test-session"})

    mock_generate.assert_called_once()
    assert result["disclaimer"] is not None
