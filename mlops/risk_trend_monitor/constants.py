"""Shared constants for the Risk-Trend tool and its drift monitor.

Pulled out on its own to avoid a circular import between tools/credit_tools.py
(which runs inference) and mlops/risk_trend_monitor/drift_monitor.py (which checks
the inputs before inference runs).
"""

# Must match RISK_TREND_FEATURES in finbuddy-project's evaluate_classification.py —
# delta_<upi_signal> for each of the 8 UPI signals. Kept in sync manually since this
# project only reads that repo's artifact, never imports its code.
RISK_TREND_FEATURE_ORDER = [
    "delta_avg_monthly_income",
    "delta_income_regularity_score",
    "delta_tx_count_30d",
    "delta_merchant_diversity",
    "delta_balance_dip_frequency",
    "delta_b2b_ratio",
    "delta_avg_transaction_size",
    "delta_tenure_months",
]
