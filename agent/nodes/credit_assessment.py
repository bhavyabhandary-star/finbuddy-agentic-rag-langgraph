"""Credit-assessment node: calls the two credit-assessment tools.

Both tools are read-only consumers of production FinBuddy's models — see
build_prompt.md. This node does not call the Generate/LLM node's provider at all;
it only invokes the two tools and lets Generate turn the result into a plain-English
answer afterward.
"""
from __future__ import annotations

from agent.state import AgentState
from observability.tracing import RunTracer
from tools.credit_tools import assess_credit_profile, assess_risk_trend


def credit_assessment_node(
    state: AgentState, signals: dict, delta_features: dict | None = None, tracer: RunTracer | None = None
) -> AgentState:
    credit_result = assess_credit_profile(signals)
    risk_trend_result = assess_risk_trend(delta_features) if delta_features else None

    if tracer:
        tracer.log_step(
            "credit_assessment",
            decision="called assess_credit_profile" + (" + assess_risk_trend" if delta_features else ""),
            tool_error=credit_result.tool_error,
        )

    return {
        **state,
        "credit_assessment": credit_result,
        "risk_trend": risk_trend_result,
        "sufficient": True,  # no retrieval loop on this path
    }
