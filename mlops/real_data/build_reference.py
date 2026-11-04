"""Summarise the SYNTHETIC data the credit models were trained on into per-signal quantiles.

Read-only on production FinBuddy's dataset; writes a small JSON next to this file so the shift report
needs no access to the production repo at run time (the same pattern as
mlops/risk_trend_monitor/reference_distribution.json).

Run: python -m mlops.real_data.build_reference "<path>/scoring_service/data/synthetic_upi_dataset.csv"
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from mlops.real_data.constants import SIGNALS

LEVELS = [0, 1, 5, 25, 50, 75, 95, 99, 100]
OUT = Path(__file__).parent / "reference_upi_distribution.json"


def main(csv_path: str) -> None:
    columns: dict[str, list[float]] = {s: [] for s in SIGNALS}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for s in SIGNALS:
                columns[s].append(float(row[s]))
    n = len(columns[SIGNALS[0]])
    reference = {
        "source": "synthetic_upi_dataset.csv (production FinBuddy scoring_service, generator-made, NOT real borrowers)",
        "n_rows": n,
        "levels": LEVELS,
        "quantiles": {s: [float(np.percentile(columns[s], q)) for q in LEVELS] for s in SIGNALS},
    }
    OUT.write_text(json.dumps(reference, indent=1), encoding="utf-8")
    print(f"wrote {OUT} from {n} synthetic rows")


if __name__ == "__main__":
    main(sys.argv[1])
