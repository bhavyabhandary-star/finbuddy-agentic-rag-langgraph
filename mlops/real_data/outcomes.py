"""Observed outcomes for real profiles. Without these nothing supervised can be trained.

data/real_outcomes.jsonl (gitignored), one JSON object per line:
  {"user_id": "...", "repaid_on_time": 0|1, "observed_at": "2027-03-01"}              -> F-001 / F-003 label
  {"user_id": "...", "trend_label": 0|1, "early_window_end": "...", "late_window_end": "..."} -> Risk-Trend label
A user may appear in both kinds of row.
"""
from __future__ import annotations

import json
from pathlib import Path

from config import settings


def load_outcomes(path: str | Path | None = None) -> list[dict]:
    target = Path(path) if path is not None else Path(settings.REAL_OUTCOMES_PATH)
    if not target.exists():
        return []
    rows = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def repayment_labels(outcomes: list[dict]) -> dict[str, int]:
    return {str(o["user_id"]): int(o["repaid_on_time"]) for o in outcomes if o.get("repaid_on_time") in (0, 1)}


def trend_labels(outcomes: list[dict]) -> dict[str, int]:
    return {str(o["user_id"]): int(o["trend_label"]) for o in outcomes if o.get("trend_label") in (0, 1)}
