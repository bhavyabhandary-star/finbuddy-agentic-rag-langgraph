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

    Timeout is 60s, not a tighter number — found necessary by hitting this
    live: the free-tier HF Space this API runs on sleeps after inactivity and
    took ~60-90s to cold-start on a real call. A 5s timeout meant the FIRST
    real request after any idle period always failed and escalated — a bad
    first impression for a demo, not a safety margin worth keeping tight.
    """

    def _call() -> CreditAssessmentResult:
        payload = dict(signals)
        if geography:
            payload["geography"] = geography
        with httpx.Client(timeout=60.0) as client:
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
    """Runs the read-only Risk-Trend (Logistic Regression, Ridge/L2) artifact.

    CONTRACT (verified against finbuddy-project's actual training code, not
    assumed): delta_features must be PRE-Z-SCORED against the training
    population, not raw natural-unit deltas (e.g. rupees). The training
    pipeline (generate_synthetic_upi_data.generate_risk_trend_dataset) fits on
    `_zscore(late - early)` per signal, and that scaler was never persisted as
    its own artifact — it only exists implicitly inside that one-off script.
    This project deliberately matches production FinBuddy's own scoping here:
    production itself never exposes this model to a raw-delta caller either —
    it only runs as an internal monthly batch job over its own regenerated
    feature pipeline (see the capstone deck's "Risk-Trend classifier" slide).
    Whoever populates `risk_trend_delta_features` (the API layer, per
    agent/state.py's note) is responsible for producing already-scaled deltas.

    Two of the 8 declared features — delta_avg_transaction_size,
    delta_tenure_months — are constant-zero in 100% of the real training data
    (verified against the real synthetic_risk_trend_dataset.csv) and the
    loaded model's own learned coefficients are exactly 0.0000 for both.
    Whatever value is passed for them here has zero effect on the prediction.
    """
    drift_flag = check_input_drift(delta_features)

    def _call() -> RiskTrendResult:
        import pandas as pd

        model = _load_risk_trend_model()
        # A plain list-of-lists produced a real sklearn UserWarning ("X does
        # not have valid feature names, but LogisticRegression was fitted
        # with feature names") — the training pipeline fit on a named
        # DataFrame (train_df[DELTA_FEATURE_COLUMNS]), so matching that shape
        # exactly here, not just the column order, is what the model expects.
        ordered = pd.DataFrame([delta_features], columns=RISK_TREND_FEATURE_ORDER)
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
