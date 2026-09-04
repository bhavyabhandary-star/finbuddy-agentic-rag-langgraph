"""Input-drift detection (PSI) for the read-only Risk-Trend tool.

Detect-only, per build_prompt.md: this project consumes production FinBuddy's
Risk-Trend artifact but doesn't own retraining it. A model trained on synthetic
data, fed differently-shaped inputs from this new project, is exactly the kind of
silent-failure case monitoring exists to catch — so this flags drift and lets the
caller fall back to assess_credit_profile's own is_anomalous signal instead.

Mirrors the PSI technique documented in finbuddy-project's
scoring_service/monitoring/drift_report.py (pure numpy, not the `evidently`
package — that repo's own README explains why: an unverifiable dependency on the
dev machine). This project reimplements the technique, not imports that file.
"""
from __future__ import annotations

import json
import os

import numpy as np

from mlops.risk_trend_monitor.constants import RISK_TREND_FEATURE_ORDER

PSI_AMBER_THRESHOLD = 0.10
PSI_RED_THRESHOLD = 0.20

# TODO(milestone step 7): populate this from the actual synthetic training
# distribution used by finbuddy-project's generate_risk_trend_dataset() — for now
# this is an empty placeholder so the function is safe to call before that
# reference distribution has been captured.
_REFERENCE_DISTRIBUTION_PATH = os.environ.get(
    "RISK_TREND_REFERENCE_DISTRIBUTION_PATH",
    "./mlops/risk_trend_monitor/reference_distribution.json",
)


def _population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Standard PSI: bins the reference distribution, compares current's bin shares."""
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(reference, quantiles)
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_input_drift(delta_features: dict) -> bool:
    """Returns True (drift flag) if delta_features look off-distribution, or if no
    reference distribution has been captured yet (fail toward caution, not silence).
    """
    if not os.path.exists(_REFERENCE_DISTRIBUTION_PATH):
        return True  # no baseline yet — flag rather than silently assume it's fine

    with open(_REFERENCE_DISTRIBUTION_PATH) as f:
        reference = json.load(f)

    flags = []
    for feature in RISK_TREND_FEATURE_ORDER:
        ref_values = np.array(reference.get(feature, []))
        if ref_values.size == 0:
            flags.append(True)
            continue
        current_value = np.array([delta_features[feature]])
        psi = _population_stability_index(ref_values, current_value)
        flags.append(psi >= PSI_AMBER_THRESHOLD)

    return any(flags)
