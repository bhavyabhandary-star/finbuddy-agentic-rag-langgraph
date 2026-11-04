"""Write the joined real training table the production trainers can consume -- ONLY if the gate passes.

The production trainers (scoring_service/models/train_credit_score.py and friends) read a CSV of the 8 signals
plus a label. This produces that file from real, consented profiles joined to observed outcomes, so retraining
is a pointer change, not a rewrite. It refuses (raises NotEligible) while the readiness gate says BLOCKED, so a
half-empty "real" table can never be mistaken for training data.
"""
from __future__ import annotations

import csv
from pathlib import Path

from config import settings
from mlops.real_data.constants import ORIGINS_PRODUCTION, SIGNALS
from mlops.real_data.outcomes import load_outcomes, repayment_labels
from mlops.real_data.profiles import load_real_profiles
from mlops.real_data.readiness import assess


class NotEligible(RuntimeError):
    pass


def export_f001_training_table(path: Path | None = None, loaded=None, outcomes=None) -> Path:
    loaded = loaded if loaded is not None else load_real_profiles()
    outcomes = outcomes if outcomes is not None else load_outcomes()
    f001 = assess(loaded, outcomes)["models"][0]
    if not f001["eligible"]:
        raise NotEligible("F-001 is not eligible for real-data retraining: " + "; ".join(f001["blockers"]))
    labels = repayment_labels(outcomes)
    target = Path(path) if path is not None else settings.REAL_TRAINING_TABLE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", *SIGNALS, "repaid_on_time"])
        for p in loaded.valid:
            uid = str(p["user_id"])
            if p.get("source") in ORIGINS_PRODUCTION and uid in labels:
                w.writerow([uid, *[p[s] for s in SIGNALS], labels[uid]])
    return target
