"""assess_credit_profile and assess_risk_trend — the two credit-assessment tools.

Per build_prompt.md: these call FinBuddy's real, already-governed production
models rather than reimplementing them. assess_credit_profile is a read-only HTTP
call to the live scoring API (F-006/F-001/F-003/F-012 together); assess_risk_trend
loads the registered Risk-Trend artifact read-only and runs inference locally.
"""
from __future__ import annotations

import os

import httpx
import joblib

from guardrails.tool_policy import safe_tool_call
from mlops.risk_trend_monitor.constants import RISK_TREND_FEATURE_ORDER
from mlops.risk_trend_monitor.drift_monitor import check_input_drift
from tools.schemas import CreditAssessmentResult, RiskTrendResult, ScoreFactor

SCORING_API_BASE_URL = os.environ.get(
    "FINBUDDY_SCORING_API_BASE_URL", "https://bhavyabhandary-finbuddy-scoring.hf.space"
)
RISK_TREND_ARTIFACT_PATH = os.environ.get(
    "RISK_TREND_ARTIFACT_PATH", "./mlops/risk_trend_monitor/artifacts/risk_trend_logreg.joblib"
)


def assess_credit_profile(signals: dict, geography: str | None = None) -> CreditAssessmentResult:
    """Read-only call to production FinBuddy's POST /api/v1/score.

    Never forwards these signals anywhere else — see guardrails/tool_policy.py's
    tool-call policy and kickoff_prompt.md's guardrail table.
    """

    def _call() -> CreditAssessmentResult:
        payload = dict(signals)
        if geography:
            payload["geography"] = geography
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(f"{SCORING_API_BASE_URL}/api/v1/score", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return CreditAssessmentResult(
            credit_score=data["credit_score"],
            calibrated_probability_of_repayment=data["calibrated_probability_of_repayment"],
            approved=data["approved"],
            fairness_mitigation_applied=data["fairness_mitigation_applied"],
            income_band=data["income_band"],
            is_anomalous=data["is_anomalous"],
            anomaly_note=data.get("anomaly_note"),
            top_3_factors=[ScoreFactor(**f) for f in data["top_3_factors"]],
            latency_ms=data["latency_ms"],
        )

    return safe_tool_call(
        _call,
        tool_name="assess_credit_profile",
        fallback=lambda err: CreditAssessmentResult(
            credit_score=0,
            calibrated_probability_of_repayment=0.0,
            approved=False,
            fairness_mitigation_applied=False,
            income_band="unknown",
            is_anomalous=False,
            anomaly_note=None,
            top_3_factors=[],
            latency_ms=0.0,
            tool_error=str(err),
        ),
    )


_risk_trend_model = None  # lazy-loaded, see _load_risk_trend_model()


def _load_risk_trend_model():
    global _risk_trend_model
    if _risk_trend_model is None:
        # Read-only load of a file copied from finbuddy-project's artifacts dir —
        # this project never imports or edits that repo's code.
        _risk_trend_model = joblib.load(RISK_TREND_ARTIFACT_PATH)
    return _risk_trend_model


def assess_risk_trend(delta_features: dict) -> RiskTrendResult:
    """Runs the read-only Risk-Trend (Logistic Regression Ridge/Lasso) artifact.

    delta_features must contain all keys in RISK_TREND_FEATURE_ORDER — same
    contract as finbuddy-project's train_risk_trend.py.
    """
    drift_flag = check_input_drift(delta_features)

    def _call() -> RiskTrendResult:
        model = _load_risk_trend_model()
        ordered = [[delta_features[k] for k in RISK_TREND_FEATURE_ORDER]]
        proba_improving = float(model.predict_proba(ordered)[0][1])
        trend = "improving" if proba_improving >= 0.5 else "decaying"
        return RiskTrendResult(
            trend=trend, probability=proba_improving, input_drift_flag=drift_flag
        )

    return safe_tool_call(
        _call,
        tool_name="assess_risk_trend",
        fallback=lambda err: RiskTrendResult(
            trend="unknown", probability=0.5, input_drift_flag=True
        ),
    )
