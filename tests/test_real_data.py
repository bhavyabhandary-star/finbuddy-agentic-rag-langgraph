"""The real-data readiness gate: it must refuse early, count only data that can justify a lending
decision, and never let a mostly-empty 'real' table be exported as training data."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from config import settings
from mlops.real_data.collect import register_consent
from mlops.real_data.constants import SIGNALS
from mlops.real_data.export import NotEligible, export_f001_training_table
from mlops.real_data.profiles import LoadedProfiles, load_real_profiles, validate_profile
from mlops.real_data.readiness import assess
from mlops.real_data.shift import shift_report

GOOD = {
    "user_id": "u1",
    "source": "setu_aa_sandbox",
    "avg_monthly_income": 20000.0,
    "income_regularity_score": 0.8,
    "tx_count_30d": 40,
    "merchant_diversity": 10,
    "balance_dip_frequency": 2,
    "b2b_ratio": 0.1,
    "avg_transaction_size": 300.0,
    "tenure_months": 6,
}


def profile(uid: str, source: str = "setu_aa") -> dict:
    return {**GOOD, "user_id": uid, "source": source}


def loaded_with(profiles: list[dict]) -> LoadedProfiles:
    return LoadedProfiles(valid=profiles)


# ---------------------------------------------------------------- validation


def test_a_good_profile_validates():
    assert validate_profile(GOOD) == []


@pytest.mark.parametrize(
    "bad",
    [
        {"source": "somewhere_else"},
        {"avg_monthly_income": -5},
        {"income_regularity_score": 1.7},
        {"tx_count_30d": None},
        {"b2b_ratio": "high"},
        {"user_id": ""},
    ],
)
def test_bad_profiles_are_rejected_with_a_reason(bad):
    assert validate_profile({**GOOD, **bad})


def test_loading_rejects_bad_lines_and_drops_duplicates(tmp_path):
    path = tmp_path / "profiles.jsonl"
    rows = [GOOD, GOOD, {**GOOD, "user_id": "u2", "b2b_ratio": 9}, "not json"]
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    loaded = load_real_profiles(path)
    assert [p["user_id"] for p in loaded.valid] == ["u1"]
    assert loaded.duplicates == 1 and len(loaded.rejected) == 2


def test_a_missing_file_is_an_empty_inventory_not_an_error(tmp_path):
    assert load_real_profiles(tmp_path / "nope.jsonl").valid == []


# ---------------------------------------------------------------- shift

REFERENCE = {
    "source": "test",
    "n_rows": 100,
    "levels": [0, 1, 5, 25, 50, 75, 95, 99, 100],
    "quantiles": {s: [0, 1, 2, 3, 4, 5, 6, 7, 8] for s in SIGNALS},
}


def test_shift_flags_values_outside_the_training_range_and_band():
    p = {**GOOD, **{s: 4 for s in SIGNALS}, "avg_monthly_income": 50, "tx_count_30d": 7.5}
    report = shift_report([p], REFERENCE)
    row = report["profiles"][0]
    assert "avg_monthly_income" in row["signals_outside_training_range"]
    assert "tx_count_30d" in row["signals_outside_band"] and "tx_count_30d" not in row["signals_outside_training_range"]
    assert row["per_signal"]["merchant_diversity"]["outside_band"] is False
    assert report["share_outside_band_by_signal"]["avg_monthly_income"] == 1.0


# ---------------------------------------------------------------- readiness gate


def test_the_current_situation_is_blocked_everywhere_and_says_why():
    report = assess(loaded_with([profile("u1", "setu_aa_sandbox")]), [])
    assert not report["any_eligible"]
    assert all(not m["eligible"] for m in report["models"])
    assert report["inventory"]["sandbox_profiles"] == 1 and report["inventory"]["production_consented_profiles"] == 0
    assert any("SANDBOX" in n for n in report["notes"])


def test_outcomes_on_sandbox_profiles_do_not_count():
    profiles = [profile(f"s{i}", "setu_aa_sandbox") for i in range(1200)]
    outcomes = [{"user_id": f"s{i}", "repaid_on_time": i % 2} for i in range(1200)]
    report = assess(loaded_with(profiles), outcomes)
    assert not report["any_eligible"]
    assert report["inventory"]["outcomes_ignored_because_profile_is_sandbox_or_unknown"] == 1200


def test_enough_production_data_with_outcomes_makes_f001_f003_f006_eligible_but_not_risk_trend():
    profiles = [profile(f"p{i}") for i in range(1000)]
    outcomes = [{"user_id": f"p{i}", "repaid_on_time": 1 if i % 10 else 0} for i in range(1000)]  # 100 minority
    report = assess(loaded_with(profiles), outcomes)
    verdicts = {m["model"].split(" ")[0]: m["eligible"] for m in report["models"]}
    assert verdicts == {"F-001": True, "F-003": True, "F-006": True, "Risk-Trend": False}


def test_a_lopsided_outcome_split_is_not_enough_even_with_many_rows():
    profiles = [profile(f"p{i}") for i in range(1200)]
    outcomes = [{"user_id": f"p{i}", "repaid_on_time": 1 if i >= 20 else 0} for i in range(1200)]  # 20 minority
    f001 = assess(loaded_with(profiles), outcomes)["models"][0]
    assert not f001["eligible"] and any("minority" in b for b in f001["blockers"])


def test_risk_trend_needs_its_own_trend_labels():
    profiles = [profile(f"p{i}") for i in range(1000)]
    outcomes = [{"user_id": f"p{i}", "trend_label": i % 2} for i in range(1000)]
    rt = assess(loaded_with(profiles), outcomes)["models"][3]
    assert rt["eligible"]


# ---------------------------------------------------------------- export refuses while blocked


def test_export_refuses_while_blocked_and_writes_nothing(tmp_path):
    target = tmp_path / "table.csv"
    with pytest.raises(NotEligible):
        export_f001_training_table(target, loaded_with([profile("u1", "setu_aa_sandbox")]), [])
    assert not target.exists()


def test_export_writes_only_labelled_production_rows_once_eligible(tmp_path):
    profiles = [profile(f"p{i}") for i in range(1000)] + [profile("sandbox1", "setu_aa_sandbox")]
    outcomes = [{"user_id": f"p{i}", "repaid_on_time": 1 if i % 4 else 0} for i in range(1000)]
    outcomes.append({"user_id": "sandbox1", "repaid_on_time": 1})
    target = export_f001_training_table(tmp_path / "t.csv", loaded_with(profiles), outcomes)
    rows = list(csv.DictReader(open(target, encoding="utf-8")))
    assert len(rows) == 1000 and "sandbox1" not in {r["user_id"] for r in rows}
    assert list(rows[0].keys()) == ["user_id", *SIGNALS, "repaid_on_time"]


# ---------------------------------------------------------------- consent register + hygiene


def test_the_consent_register_stores_a_hash_never_the_mobile(tmp_path):
    reg = tmp_path / "register.jsonl"
    entry = register_consent("9999999999@onemoney", GOOD, 12, "validation", reg)
    text = reg.read_text()
    assert "9999999999" not in text and entry["vua_sha256"] in text and entry["retain_until"] > entry["collected_at"]


def test_personal_data_files_are_gitignored():
    ignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text()
    for pattern in ("data/real_*.jsonl", "data/real_*.csv", "data/setu_real_profiles.jsonl"):
        assert pattern in ignore


def test_thresholds_have_not_been_quietly_lowered():
    assert (settings.REAL_MIN_LABELLED_F001, settings.REAL_MIN_MINORITY_CLASS) == (1000, 100)
    assert settings.REAL_MIN_PROFILES_F006 == 500 and settings.REAL_MIN_TREND_USERS == 1000


def test_preflight_reports_missing_setu_settings_by_name_only():
    from mlops.real_data.preflight import check

    r = check({"SETU_BASE_URL": "https://fiu-sandbox.setu.co", "SETU_CLIENT_ID": "x"})
    assert r["ready"] is False and r["missing"] == ["SETU_CLIENT_SECRET", "SETU_PRODUCT_INSTANCE_ID"]
    assert r["looks_like_sandbox"] is True
    full = check({n: "v" for n in ("SETU_BASE_URL", "SETU_CLIENT_ID", "SETU_CLIENT_SECRET", "SETU_PRODUCT_INSTANCE_ID")})
    assert full["ready"] is True and full["looks_like_sandbox"] is False
