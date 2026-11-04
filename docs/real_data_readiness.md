# Real-data readiness: retraining F-001 / F-003 / F-006 / Risk-Trend on real, consented Setu AA data

Production FinBuddy's data generator states the condition plainly: its reported AUCs are "a statement about this
generator", and the models "must be retrained on real, consented Setu Account Aggregator data" before any real
lending decision relies on them. This page is the honest status of that condition and the machinery built to meet it.

## Status (2026-11-02)

**Retraining is not yet permitted, and none was done.** The gate (`python -m mlops.real_data.readiness`) says BLOCKED for all four models:

| Model | Needs, from real consented borrowers | Held today | Status |
|---|---|---|---|
| F-001 credit score (XGBoost) | 1,000 profiles with an observed repayment outcome, at least 100 in the minority class | 0 | Blocked |
| F-003 SHAP explainer | Rebuilt on the retrained F-001, so it needs the same data | 0 | Blocked |
| F-006 anomaly (Isolation Forest) | 500 profiles; no labels needed | 0 (4 sandbox profiles) | Blocked |
| Risk-Trend (logistic regression) | 1,000 users observed in two windows with an observed trend label, at least 100 minority | 0 | Blocked |

Why the thresholds: with 1,000 labelled profiles and at least 100 minority outcomes, the AUC of a held-out 20% split has a
standard error of roughly 0.03, enough to tell a real improvement from noise. They are in `config/settings.py`
(`REAL_MIN_*`) and a test fails if someone quietly lowers them.

## Two reasons the data we hold cannot be used

1. **It is sandbox data.** The four profiles we hold came from live pulls on Setu's AA sandbox: the real AA protocol and a
   real consent flow, but Setu's *mock* bank accounts, from one vendor and one phone number. They validate the plumbing.
   They say nothing about how real borrowers behave, so the gate excludes sandbox data from every threshold. Pulling more
   sandbox profiles would not change this.
2. **There are no outcomes.** F-001 and Risk-Trend are supervised. They need to know who repaid and whose trend improved,
   observed after the profile was taken. No profile we hold has one, and outcomes cannot be created: they happen over months.

## What the sandbox profiles already show (`python -m mlops.real_data.shift`)

Compared with the 10,000 synthetic rows the models were trained on, **each of the four sandbox profiles** is outside the
training range on the same 3 of 8 signals and outside the 1st-99th percentile band on a 4th:

| Signal | Sandbox values (4 profiles) | Synthetic range |
|---|---|---|
| avg_monthly_income | ₹81,938 to ₹128,420 | max ₹73,157 (above) |
| tx_count_30d | 9 to 13 | min 24 (below) |
| b2b_ratio | 0.0 | min 0.0009 (at the floor) |
| avg_transaction_size | ₹23,248 to ₹28,567 | p99 ₹1,695; about 14 to 17x the 99th percentile |

The synthetic data looks like small, frequent UPI spending; Setu's deposit-account data looks like salary-sized,
infrequent bank-account activity. So the scoring models were never shown inputs like this, and their output on them is
extrapolation. It is consistent with the "unusual pattern" flag the scoring API raised for such a profile in the live API
run. Four profiles from one mock-data generator say how well the features fit that data, not what real borrowers look like.

## Live run, 2026-11-05

A consent was created through the API, approved by a person in Setu's UAT webview, and the data fetched: 4 of 6 shared
accounts delivered (the 2 `FAILUREXXXXX` test accounts time out by design, and Setu then marks the whole session
`FAILED`, which the client treats as usable when any account is delivered). The bank payload was reduced to the 8 signals,
scored (683, flagged unusual), and the mobile was absent from the response and from every trace file. A consent is
`ONETIME`: one data fetch, so an interrupted run needs a fresh approval (`collect --consent-id --session-id` resumes a
started session).

## What was built (all in `mlops/real_data/`, agentic repo only)

| Piece | Does |
|---|---|
| `profiles.py` | Loads real profiles, validates the 8 signals against plausible ranges, rejects bad lines, drops duplicates, tracks sandbox vs production origin |
| `outcomes.py` | Joins observed repayment and trend outcomes (`data/real_outcomes.jsonl`; none exist yet) |
| `shift.py` | Reports where each real value falls in the synthetic training distribution |
| `readiness.py` | The gate above: counts only production-consented data, reports exactly what is missing |
| `export.py` | Writes the joined real training table in the shape the production trainers read, and **refuses while the gate is BLOCKED** |
| `collect.py` | Pulls a fresh consented profile and writes a consent-register entry (hashed mobile, purpose, retention date) |

Personal data files (`data/real_*.jsonl`, `data/real_*.csv`, `data/setu_real_profiles.jsonl`) are gitignored; a test checks that.
Tests: `tests/test_real_data.py`. No production model was modified or retrained, and the production repo was not touched.

## What it takes to get to real retraining

1. **Real consented borrowers, through a regulated FIU.** On Setu's AA gateway an entity must be regulated and licensed (a bank, NBFC
   or similar) to go live as a Financial Information User; FinBuddy, as an unlicensed product, cannot be one itself. The route is a
   regulated partner that is the FIU, with FinBuddy as its technology and scoring provider under a data-processing agreement, a
   DPDP-compliant consent flow, a stated purpose and a retention limit. A sandbox cannot supply this.
2. **Outcomes.** Track repayment for 6 to 12 months and record it in `data/real_outcomes.jsonl`.
3. **Run the gate.** When it reports ELIGIBLE, export the table (`mlops.real_data.export`), point the production trainers at it,
   and retrain in the production repo.
4. **Same governance as the router.** The retrained model must beat the current one on a real held-out set, the fairness audit
   (F-012) must be re-run on real data, and a named human signs it off before it serves.

Until then, every score this system returns is a decision-support illustration, and the disclaimer on each credit answer says so.
