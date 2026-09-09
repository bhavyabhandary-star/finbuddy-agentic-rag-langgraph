"""Input-drift/out-of-range detection for the read-only Risk-Trend tool.

Detect-only, per build_prompt.md: this project consumes production FinBuddy's
Risk-Trend artifact but doesn't own retraining it. A model trained on synthetic
data, fed differently-shaped inputs from this new project, is exactly the kind of
silent-failure case monitoring exists to catch — so this flags drift and lets the
caller fall back to assess_credit_profile's own is_anomalous signal instead.

Two checks, for two different jobs — found necessary by actually testing this,
not assumed upfront:
  - check_input_drift(): a PER-REQUEST out-of-range check (is this one value a
    plausible sample from the reference distribution?) — uses percentile
    bounds, not PSI.
  - fit_reference_bins() / _population_stability_index_from_bins(): PSI,
    mirroring the technique in finbuddy-project's
    scoring_service/monitoring/drift_report.py (pure numpy, not `evidently` —
    that repo's own README explains why). PSI is a BATCH-comparison
    statistic — verified directly that applying it to a single incoming value
    (n=1) against a 10-bin reference always reports "drift", regardless of
    whether the value is typical, because one point concentrating 100% of its
    mass in a single bin against a reference spread ~10%/bin is guaranteed to
    diverge. Kept here for a future weekly-batch dashboard (per
    build_prompt.md's drift-ladder design), not used per-request.
"""
from __future__ import annotations

import json
import os

import numpy as np

from mlops.risk_trend_monitor.constants import RISK_TREND_FEATURE_ORDER

PSI_AMBER_THRESHOLD = 0.10
PSI_RED_THRESHOLD = 0.20
PSI_BINS = 10

# Built from the real synthetic_risk_trend_dataset.csv (10,000 rows) that
# finbuddy-project's generate_risk_trend_dataset() produced — see
# mlops/risk_trend_monitor/build_reference_distribution.py. Stores only the
# per-feature percentile breakpoints + reference bin shares (~20 floats per
# feature), not the 10,000 raw rows (1.3MB) — PSI only ever needs the bin
# edges and the reference proportions, never the raw values themselves.
_REFERENCE_DISTRIBUTION_PATH = os.environ.get(
    "RISK_TREND_REFERENCE_DISTRIBUTION_PATH",
    "./mlops/risk_trend_monitor/reference_distribution.json",
)


def fit_reference_bins(reference: np.ndarray, bins: int = PSI_BINS) -> dict:
    """One-time fit: percentile breakpoints + reference bin shares for one feature.
    Called by build_reference_distribution.py, not at request time.

    Two of the 8 declared features (delta_avg_transaction_size,
    delta_tenure_months) are constant-zero in every training row — verified
    directly against the real synthetic_risk_trend_dataset.csv and against the
    loaded model's own coefficients (both exactly 0.0000). PSI is undefined
    for a zero-variance reference (percentile breakpoints all collapse to the
    same value, which np.histogram rejects as non-monotonic), and moot anyway
    since the model mathematically ignores these two — marked "constant"
    instead of fit, and never flagged for drift.
    """
    if float(np.std(reference)) == 0.0:
        return {"constant": True, "value": float(reference[0])}

    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(reference, quantiles)
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)

    # p01/p99 for the per-request out-of-range check (check_input_drift) —
    # a separate, looser band than the PSI deciles above: flagging outside the
    # 10th-90th percentile would trigger on ~20% of genuinely typical values,
    # which isn't a meaningful "does this look wrong" signal for one request.
    p01, p99 = np.percentile(reference, [1, 99])

    return {
        "breakpoints": breakpoints.tolist(),
        "ref_pct": ref_pct.tolist(),
        "p01": float(p01),
        "p99": float(p99),
    }


def _population_stability_index_from_bins(breakpoints: list, ref_pct: list, current_value: float) -> float:
    """Scores one new value against pre-fit reference bins (see fit_reference_bins)."""
    cur_counts, _ = np.histogram([current_value], bins=breakpoints)
    cur_pct = np.clip(cur_counts / 1, 1e-6, None)
    ref_pct = np.clip(np.array(ref_pct), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_input_drift(delta_features: dict) -> bool:
    """Per-request check: is each value a plausible sample from the reference
    distribution (within its 1st-99th percentile band)? Returns True (drift
    flag) if any feature is out of range, or if no reference distribution has
    been captured yet (fail toward caution, not silence).

    Deliberately NOT PSI — see module docstring for why PSI on a single value
    always reports drift regardless of whether the value is typical.
    """
    if not os.path.exists(_REFERENCE_DISTRIBUTION_PATH):
        return True  # no baseline yet — flag rather than silently assume it's fine

    with open(_REFERENCE_DISTRIBUTION_PATH) as f:
        reference = json.load(f)

    for feature in RISK_TREND_FEATURE_ORDER:
        fitted = reference.get(feature)
        if not fitted:
            return True
        if fitted.get("constant"):
            continue  # model coefficient is 0.0 for this feature — out-of-range here can't affect the prediction
        value = delta_features[feature]
        if value < fitted["p01"] or value > fitted["p99"]:
            return True

    return False
