import os

import pytest

from ingestion.setu_feed import CACHED_PROFILE_PATH, load_cached_real_profile

requires_cached_profile = pytest.mark.skipif(
    not os.path.exists(CACHED_PROFILE_PATH),
    reason="requires data/setu_real_profiles.jsonl copied read-only from finbuddy-project",
)


@requires_cached_profile
def test_load_cached_real_profile_returns_real_upi_signals():
    profile = load_cached_real_profile()
    assert profile is not None
    assert profile["source"] == "setu_aa_sandbox"
    for key in (
        "avg_monthly_income",
        "income_regularity_score",
        "tx_count_30d",
        "merchant_diversity",
        "balance_dip_frequency",
        "b2b_ratio",
        "avg_transaction_size",
        "tenure_months",
    ):
        assert key in profile


def test_load_cached_real_profile_returns_none_when_missing(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.jsonl")
    assert load_cached_real_profile(missing_path) is None
