"""Typed contracts for every tool result and the final agent response.

Forcing structured output (kickoff_prompt.md's "Structured output & validation"
section) turns "parse the model's prose" into "validate a structured object."
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float = Field(..., description="Cosine similarity or hybrid rank score")


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    top_score: float
    sufficient: bool = Field(
        ..., description="Classical threshold check result — never an LLM self-judgment"
    )


class ScoreFactor(BaseModel):
    """Mirrors production FinBuddy's ScoreFactor shape (scoring_service/api/main.py)."""

    factor: str
    shap_contribution: float
    direction: str
    plain_english: str
    action: str


class CreditAssessmentResult(BaseModel):
    """Passthrough of production FinBuddy's /api/v1/score response.

    F-006 (anomaly), F-001 (score), F-003 (SHAP top-3), F-012 (fairness-mitigated
    decision) all arrive together in this one call — see build_prompt.md.
    """

    credit_score: int
    calibrated_probability_of_repayment: float
    approved: bool
    fairness_mitigation_applied: bool
    income_band: str
    is_anomalous: bool
    anomaly_note: str | None
    top_3_factors: list[ScoreFactor]
    latency_ms: float
    tool_error: str | None = Field(
        None, description="Set when the live scoring API call failed — see guardrails/tool_policy.py"
    )


class RiskTrendResult(BaseModel):
    """Output of the read-only Risk-Trend (Logistic Regression Ridge/Lasso) tool."""

    trend: str = Field(..., description='"improving" or "decaying"')
    probability: float
    input_drift_flag: bool = Field(
        False, description="Set by mlops/risk_trend_monitor if input features look off-distribution"
    )


class AgentResponse(BaseModel):
    """The Generate node's forced-JSON output contract."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float
    escalate_to_human: bool = False
