import os
from unittest.mock import MagicMock, patch

import pytest

from tools.credit_tools import RISK_TREND_ARTIFACT_PATH, assess_credit_profile, assess_risk_trend
from tools.rag_tools import check_sufficiency

requires_risk_trend_artifact = pytest.mark.skipif(
    not os.path.exists(RISK_TREND_ARTIFACT_PATH),
    reason="requires risk_trend_logreg.joblib copied read-only from finbuddy-project — see README setup",
)

# Real rows sampled from finbuddy-project's synthetic_risk_trend_dataset.csv
# (already z-scored, matching this tool's documented contract) — hardcoded so
# this test doesn't need cross-repo file access to run.
_REAL_IMPROVING_ROW = {
    "delta_avg_monthly_income": 0.553392,
    "delta_income_regularity_score": 1.68956,
    "delta_tx_count_30d": 0.94078,
    "delta_merchant_diversity": 0.926998,
    "delta_balance_dip_frequency": -1.809724,
    "delta_b2b_ratio": 0.191642,
    "delta_avg_transaction_size": 0.0,
    "delta_tenure_months": 0.0,
}
_REAL_DECAYING_ROW = {
    "delta_avg_monthly_income": -0.226141,
    "delta_income_regularity_score": -0.778607,
    "delta_tx_count_30d": -0.678112,
    "delta_merchant_diversity": -1.022484,
    "delta_balance_dip_frequency": 0.306008,
    "delta_b2b_ratio": -0.600829,
    "delta_avg_transaction_size": 0.0,
    "delta_tenure_months": 0.0,
}


def test_check_sufficiency_above_threshold():
    assert check_sufficiency(0.85) is True


def test_check_sufficiency_below_threshold():
    assert check_sufficiency(0.30) is False


def test_assess_credit_profile_success():
    fake_response = {
        "credit_score": 738,
        "calibrated_probability_of_repayment": 0.81,
        "approved": True,
        "fairness_mitigation_applied": True,
        "income_band": "middle",
        "is_anomalous": False,
        "anomaly_note": None,
        "top_3_factors": [],
        "latency_ms": 85.0,
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        result = assess_credit_profile({"avg_monthly_income": 21000})

    assert result.credit_score == 738
    assert result.tool_error is None


def test_assess_credit_profile_falls_back_on_error():
    with patch("httpx.Client") as mock_client_cls:
        mock_client_cls.return_value.__enter__.return_value.post.side_effect = ConnectionError("down")
        result = assess_credit_profile({"avg_monthly_income": 21000})

    assert result.tool_error is not None
    assert result.credit_score == 0


@requires_risk_trend_artifact
def test_assess_risk_trend_predicts_correctly_on_real_rows():
    """Regression test for the real bug found wiring this up: PSI applied to a
    single value always reported drift, even on rows sampled directly from
    the reference distribution — see mlops/risk_trend_monitor/drift_monitor.py.
    """
    improving = assess_risk_trend(_REAL_IMPROVING_ROW)
    assert improving.trend == "improving"
    assert improving.probability > 0.9  # real model scored this row 0.9928
    assert improving.input_drift_flag is False  # in-range data must not be flagged

    decaying = assess_risk_trend(_REAL_DECAYING_ROW)
    assert decaying.trend == "decaying"
    assert decaying.probability < 0.1  # real model scored this row 0.0277
    assert decaying.input_drift_flag is False


@requires_risk_trend_artifact
def test_assess_risk_trend_flags_out_of_range_input():
    extreme = dict(_REAL_IMPROVING_ROW)
    extreme["delta_avg_monthly_income"] = 500.0  # real z-scores here range roughly -6..+6
    result = assess_risk_trend(extreme)
    assert result.input_drift_flag is True
