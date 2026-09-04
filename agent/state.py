"""AgentState: the single state object threaded through every LangGraph node.

Short-term memory per kickoff_prompt.md IS this state, carried across the loop
and cleared per session — see memory/short_term.py.
"""
from __future__ import annotations

from typing import TypedDict

from tools.schemas import CreditAssessmentResult, RetrievedChunk, RiskTrendResult

MAX_LOOP_ITERATIONS = 3  # hard cap per kickoff_prompt.md's cost/latency section


class AgentState(TypedDict, total=False):
    # input
    query: str
    session_id: str

    # routing (mlops/intent_router)
    route: str  # "policy" | "credit_assessment" | "off_topic"
    route_confidence: float

    # RAG path
    retrieved_chunks: list[RetrievedChunk]
    top_score: float
    sufficient: bool
    loop_count: int

    # credit-assessment path — signals/delta_features are seeded by the API layer
    # (Layer 1) before the graph is invoked; the agent never collects these itself
    credit_signals: dict
    risk_trend_delta_features: dict | None
    credit_assessment: CreditAssessmentResult | None
    risk_trend: RiskTrendResult | None

    # output
    answer: str
    sources: list[str]
    confidence: float
    escalate_to_human: bool

    # guardrail / observability flags
    guardrail_flags: list[str]
