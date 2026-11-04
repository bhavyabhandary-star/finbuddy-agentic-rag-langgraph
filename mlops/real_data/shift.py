"""How far do real profiles sit from the synthetic data the models were trained on?

A model only knows the region of feature space it was trained on. This reports, per real profile and
per signal, where the real value falls in the synthetic distribution, and flags values outside the
synthetic 1st-99th percentile band (and outside its min-max range, where the model is extrapolating).
It is evidence for or against retraining -- it does not retrain anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import settings
from mlops.real_data.constants import SIGNALS

REFERENCE_PATH = Path(__file__).parent / "reference_upi_distribution.json"


def load_reference(path: Path | None = None) -> dict:
    return json.loads(Path(path or REFERENCE_PATH).read_text(encoding="utf-8"))


def percentile_of(value: float, quantiles: list[float], levels: list[int]) -> float:
    """Percentile rank (0-100) of `value` in the reference distribution; clipped to [0, 100]."""
    q = np.asarray(quantiles, dtype=float)
    # np.interp needs a non-decreasing x; repeated quantile values are fine.
    return float(np.interp(value, q, np.asarray(levels, dtype=float)))


def shift_for_profile(profile: dict, reference: dict) -> dict:
    levels = reference["levels"]
    low_i, high_i = levels.index(settings.SHIFT_BAND_LOW_PCT), levels.index(settings.SHIFT_BAND_HIGH_PCT)
    per_signal = {}
    for s in SIGNALS:
        qs = reference["quantiles"][s]
        v = float(profile[s])
        outside_band = v < qs[low_i] or v > qs[high_i]
        extrapolating = v < qs[0] or v > qs[-1]
        per_signal[s] = {
            "value": v,
            "percentile_in_synthetic": round(percentile_of(v, qs, levels), 1),
            "synthetic_band": [qs[low_i], qs[high_i]],
            "outside_band": outside_band,
            "outside_training_range": extrapolating,
        }
    return {
        "user_id": profile.get("user_id"),
        "signals_outside_band": [s for s, d in per_signal.items() if d["outside_band"]],
        "signals_outside_training_range": [s for s, d in per_signal.items() if d["outside_training_range"]],
        "per_signal": per_signal,
    }


def shift_report(profiles: list[dict], reference: dict | None = None) -> dict:
    reference = reference or load_reference()
    rows = [shift_for_profile(p, reference) for p in profiles]
    n = len(rows)
    per_signal_rate = {
        s: (sum(s in r["signals_outside_band"] for r in rows) / n if n else None) for s in SIGNALS
    }
    return {
        "n_real_profiles": n,
        "reference": {"source": reference["source"], "n_rows": reference["n_rows"]},
        "share_outside_band_by_signal": per_signal_rate,
        "profiles": rows,
    }


def format_report(report: dict) -> str:
    lines = [f"REAL vs SYNTHETIC TRAINING DISTRIBUTION  ({report['n_real_profiles']} real profile(s) vs "
             f"{report['reference']['n_rows']:,} synthetic rows)", ""]
    for p in report["profiles"]:
        lines.append(f"profile {str(p['user_id'])[:8]}...  outside the synthetic 1st-99th percentile band on "
                     f"{len(p['signals_outside_band'])} of {len(SIGNALS)} signals; outside the training range on "
                     f"{len(p['signals_outside_training_range'])}")
        for s, d in p["per_signal"].items():
            flag = "OUTSIDE RANGE" if d["outside_training_range"] else ("outside band" if d["outside_band"] else "in band")
            lines.append(f"    {s:<26} value {d['value']:>12,.2f}   synthetic percentile {d['percentile_in_synthetic']:>5.1f}   {flag}")
    return "\n".join(lines)


def main() -> int:
    import sys

    from mlops.real_data.profiles import load_real_profiles

    loaded = load_real_profiles()
    report = shift_report(loaded.valid)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
