from unittest.mock import MagicMock, patch

from tools.credit_tools import assess_credit_profile
from tools.rag_tools import check_sufficiency


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
