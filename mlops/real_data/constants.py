"""Plausibility bounds for validating real profiles (NOT model bounds: a value outside these is a
data error; a value outside the synthetic training range is a distribution shift, reported separately)."""
from __future__ import annotations

SIGNALS = (
    "avg_monthly_income",
    "income_regularity_score",
    "tx_count_30d",
    "merchant_diversity",
    "balance_dip_frequency",
    "b2b_ratio",
    "avg_transaction_size",
    "tenure_months",
)

# (min, max) inclusive
PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "avg_monthly_income": (0.0, 10_000_000.0),
    "income_regularity_score": (0.0, 1.0),
    "tx_count_30d": (0.0, 100_000.0),
    "merchant_diversity": (0.0, 100_000.0),
    "balance_dip_frequency": (0.0, 1_000.0),
    "b2b_ratio": (0.0, 1.0),
    "avg_transaction_size": (0.0, 100_000_000.0),
    "tenure_months": (0.0, 120.0),
}

# Where a profile came from. Only "production_consented" data (borrowers who consented through a
# live, registered Account Aggregator) can justify a lending decision; the Setu sandbox serves
# Setu's own mock bank accounts, so it can validate plumbing but not credit behaviour.
ORIGINS_SANDBOX = {"setu_aa_sandbox"}
ORIGINS_PRODUCTION = {"setu_aa", "setu_aa_production"}
