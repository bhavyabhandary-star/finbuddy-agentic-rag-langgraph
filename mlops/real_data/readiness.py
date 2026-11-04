"""The gate: which credit models MAY be retrained on the real data we hold, and what is missing.

This is deliberately a refusal engine. It answers "can we retrain F-001 / F-003 / F-006 / Risk-Trend on
real, consented Setu AA data?" with a number, not an opinion, and it counts only data that could justify a
lending decision:

* Sandbox profiles (Setu's mock bank accounts, real AA protocol) validate the plumbing but say nothing
  about how real borrowers behave, so they are EXCLUDED from every threshold and reported separately.
* Supervised models also need OBSERVED outcomes (did the person repay? did their trend improve?). A profile
  with no outcome cannot train F-001 or Risk-Trend, however real it is.

Run: python -m mlops.real_data.readiness [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field

from config import settings
from mlops.real_data.constants import ORIGINS_PRODUCTION
from mlops.real_data.outcomes import load_outcomes, repayment_labels, trend_labels
from mlops.real_data.profiles import LoadedProfiles, load_real_profiles


@dataclass
class ModelVerdict:
    model: str
    eligible: bool
    have: dict
    need: dict
    blockers: list[str] = field(default_factory=list)


def _production_ids(loaded: LoadedProfiles) -> set[str]:
    return {str(p["user_id"]) for p in loaded.valid if p.get("source") in ORIGINS_PRODUCTION}


def _minority(labels: dict[str, int]) -> int:
    ones = sum(labels.values())
    return min(ones, len(labels) - ones)


def assess(loaded: LoadedProfiles | None = None, outcomes: list[dict] | None = None) -> dict:
    loaded = loaded if loaded is not None else load_real_profiles()
    outcomes = outcomes if outcomes is not None else load_outcomes()
    prod_ids = _production_ids(loaded)
    all_repay, all_trend = repayment_labels(outcomes), trend_labels(outcomes)
    repay = {u: y for u, y in all_repay.items() if u in prod_ids}
    trend = {u: y for u, y in all_trend.items() if u in prod_ids}

    inventory = {
        "valid_profiles": len(loaded.valid),
        "rejected_profiles": len(loaded.rejected),
        "duplicate_profiles": loaded.duplicates,
        "sandbox_profiles": loaded.n_sandbox,
        "production_consented_profiles": loaded.n_production,
        "repayment_outcomes_on_production_profiles": len(repay),
        "trend_outcomes_on_production_profiles": len(trend),
        "outcomes_ignored_because_profile_is_sandbox_or_unknown": (len(all_repay) - len(repay)) + (len(all_trend) - len(trend)),
    }

    f001_blockers = []
    if len(repay) < settings.REAL_MIN_LABELLED_F001:
        f001_blockers.append(f"{len(repay)} labelled production-consented profiles; need {settings.REAL_MIN_LABELLED_F001}")
    if _minority(repay) < settings.REAL_MIN_MINORITY_CLASS:
        f001_blockers.append(f"minority outcome class has {_minority(repay)} rows; need {settings.REAL_MIN_MINORITY_CLASS}")
    f001 = ModelVerdict(
        "F-001 credit score (XGBoost, calibrated)",
        not f001_blockers,
        {"labelled": len(repay), "minority_class": _minority(repay)},
        {"labelled": settings.REAL_MIN_LABELLED_F001, "minority_class": settings.REAL_MIN_MINORITY_CLASS},
        f001_blockers,
    )
    f003 = ModelVerdict(
        "F-003 explainability (SHAP TreeExplainer)",
        f001.eligible,
        f001.have,
        f001.need,
        [] if f001.eligible else ["F-003 is the SHAP explainer of F-001, so it is retrained with F-001 and blocked by the same data gap"],
    )

    f006_blockers = []
    if loaded.n_production < settings.REAL_MIN_PROFILES_F006:
        f006_blockers.append(
            f"{loaded.n_production} production-consented profiles; need {settings.REAL_MIN_PROFILES_F006} (no labels required)"
        )
    f006 = ModelVerdict(
        "F-006 anomaly detection (Isolation Forest)",
        not f006_blockers,
        {"profiles": loaded.n_production},
        {"profiles": settings.REAL_MIN_PROFILES_F006},
        f006_blockers,
    )

    rt_blockers = []
    if len(trend) < settings.REAL_MIN_TREND_USERS:
        rt_blockers.append(
            f"{len(trend)} users with two observation windows and an observed trend label; need {settings.REAL_MIN_TREND_USERS}"
        )
    if _minority(trend) < settings.REAL_MIN_MINORITY_CLASS:
        rt_blockers.append(f"minority trend class has {_minority(trend)} rows; need {settings.REAL_MIN_MINORITY_CLASS}")
    risk_trend = ModelVerdict(
        "Risk-Trend classifier (logistic regression)",
        not rt_blockers,
        {"trend_labelled_users": len(trend), "minority_class": _minority(trend)},
        {"trend_labelled_users": settings.REAL_MIN_TREND_USERS, "minority_class": settings.REAL_MIN_MINORITY_CLASS},
        rt_blockers,
    )

    verdicts = (f001, f003, f006, risk_trend)
    notes = []
    if loaded.n_sandbox and not loaded.n_production:
        notes.append(
            "Every profile held is Setu SANDBOX data (mock bank accounts, real AA protocol). It validates the pipeline; "
            "it cannot justify retraining a model that a lending decision depends on."
        )
    return {
        "inventory": inventory,
        "models": [asdict(v) for v in verdicts],
        "any_eligible": any(v.eligible for v in verdicts),
        "notes": notes,
    }


def format_report(report: dict) -> str:
    lines = ["REAL-DATA READINESS (F-001 / F-003 / F-006 / Risk-Trend)", "", "Inventory"]
    for k, v in report["inventory"].items():
        lines.append(f"  {k.replace('_', ' '):<62} {v}")
    lines += ["", "Verdict per model"]
    for m in report["models"]:
        lines.append(f"  [{'ELIGIBLE' if m['eligible'] else 'BLOCKED '}] {m['model']}")
        for b in m["blockers"]:
            lines.append(f"      - {b}")
    for note in report["notes"]:
        lines += ["", "Note: " + note]
    lines += ["", "Nothing was retrained. Retraining is only permitted for models marked ELIGIBLE."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = assess()
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
