"""Builds reference_distribution.json from the real synthetic_risk_trend_dataset.csv
that finbuddy-project's generate_risk_trend_dataset() produced (10,000 rows,
already z-scored — see tools/credit_tools.py's documented contract: this tool
expects pre-z-scored deltas, matching how production FinBuddy's own internal
batch pipeline feeds this model).

Read-only consumption of that repo's data file, same as the joblib artifact —
this project never modifies or imports finbuddy-project's code.

Run: python -m mlops.risk_trend_monitor.build_reference_distribution <path-to-csv>
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from mlops.risk_trend_monitor.constants import RISK_TREND_FEATURE_ORDER
from mlops.risk_trend_monitor.drift_monitor import fit_reference_bins

OUTPUT_PATH = "mlops/risk_trend_monitor/reference_distribution.json"


def build(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    reference = {feature: fit_reference_bins(df[feature].to_numpy()) for feature in RISK_TREND_FEATURE_ORDER}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Wrote {OUTPUT_PATH} from {len(df)} rows in {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m mlops.risk_trend_monitor.build_reference_distribution <path-to-synthetic_risk_trend_dataset.csv>")
        sys.exit(1)
    build(sys.argv[1])
