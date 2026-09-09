# FinBuddy Agentic RAG — Build Prompt (ML Algorithms + MLOps)

**Read `kickoff_prompt.md` in this same folder first — in full.** It has the role
framing, the six-layer architecture, the LangGraph state machine, guardrails,
cost/latency, evaluation, and deployment sections. This document does not repeat
those; it resolves the three decisions that document left open, adds the specific
ML tools this agent orchestrates (each one mapped to the **same task it already
does in production FinBuddy** — not repurposed into something else), and adds the
MLOps this project actually owns. Once you've read both, start at the Milestone
Checklist at the bottom.

**Correction from an earlier draft**: the five capstone models below are each used
for their **original, named purpose** — credit scoring, explainability, risk-trend
classification, anomaly detection on UPI data, and bias auditing. None of them are
repurposed as a text/query classifier. Where this project genuinely needs a small
classifier for routing a user's *message* (credit-assessment request vs a
policy/coaching question vs off-topic), that is a separate, new, small component —
called the **Intent Router** below — and it is never referred to by any of the five
models' names.

---

## Resolved decisions

| Decision | Resolution |
|---|---|
| Demo corpus | Real RBI/DPDP source PDFs. Don't fabricate policy text or substitute the converted markdown corpus. |
| Local-Ollama fallback | Build it behind the same LLM interface, **verify it once with a real run** (log the output, keep proof), but don't switch to it live in the demo — same "real but not the default path" pattern the production README already uses for other optional features. Don't claim it works unless you've actually run it. |
| Deployment target | HuggingFace Spaces — a **new, separate Space**. Never touch, redeploy, or share credentials/config with the existing production FinBuddy Spaces (`bhavyabhandary-finbuddy-*`) or its repo. |

## What production FinBuddy already wires together (verified by reading the code, not assumed)

`finbuddy-project/scoring_service/api/main.py`'s live `POST /api/v1/score` endpoint
already runs, in one real call: **F-006 Isolation Forest anomaly pre-check → F-001
XGBoost calibrated credit score → F-003 SHAP top-3 factors → F-012 Fairlearn-mitigated
approval decision.** That's four of the five models, already deployed, already
governed by production FinBuddy's own MLOps (registry, drift monitor, fairness
gate — see that repo's `docs/architecture.md`). This project's job is to **call that
real endpoint as a tool**, not rebuild those four models — rebuilding them here would
be strictly worse (untested, ungoverned, duplicate) than using the real thing.

The fifth model — the **Risk-Trend Classifier** (Logistic Regression, Ridge/Lasso,
the capstone's featured deep-dive) — is trained and registered
(`scoring_service/models/artifacts/risk_trend_logreg.joblib`) but is **not** wired
into that live endpoint; production runs it as a monthly batch job, not real-time.
This project loads that artifact **read-only** to expose it as its own tool.

Neither of these counts as modifying `finbuddy-project` — reading a deployed API and
loading a trained artifact file are both read-only consumption, same as the "NBFC
lender API" consumer already named in that repo's own architecture diagram.

## ML tools this agent orchestrates (each mapped to its real, original task)

| Capstone model | Exposed here as | How this project consumes it |
|---|---|---|
| **F-001 XGBoost Gradient Boosting** — Credit Scoring | Tool: `assess_credit_profile` | Calls the **live** `POST /api/v1/score` endpoint on the production scoring Space with the user's 8 UPI signals |
| **F-003 SHAP TreeExplainer** — Explainability | Included in the same tool's response | `top_3_factors` comes back from the same `/api/v1/score` call — don't re-derive SHAP values locally, the production response already has them |
| **F-006 Isolation Forest** — Anomaly Detection | Included in the same tool's response | `is_anomalous` / `anomaly_note` come back from the same call — this is a pre-scoring check on the **UPI signals**, not on the user's chat message |
| **F-012 Fairlearn Equalized Odds** — Bias Audit | Included in the same tool's response | `fairness_mitigation_applied` and the approval decision already reflect production's fairness-mitigated threshold — surface this to the user/judge as-is, don't recompute it |
| **Logistic Regression (Ridge/Lasso)** — Risk-Trend Classifier (featured deep-dive) | Tool: `assess_risk_trend` | Loads `risk_trend_logreg.joblib` **read-only** from the production repo's artifacts directory and runs inference locally in this project — this is the one model this project actually executes itself, so it's also the one this project can meaningfully demo the Ridge-vs-Lasso engineering story around (see Demo Narrative). **Verified contract**: inputs must be pre-z-scored against the training population (the scaler was never persisted as its own artifact — it only exists implicitly inside `generate_synthetic_upi_data.py`), matching production's own scoping since it never exposes this model to a raw-delta caller either. Also verified: 2 of the 8 declared features are constant-zero in 100% of real training rows, and the model's own coefficients are exactly 0.0 for both — confirmed harmless, not guessed |

### The Intent Router (new, small, separate — not one of the five)

The agent still needs to decide, per incoming message, which path to take:

- **Credit-assessment request** → needs the user's UPI signals → call
  `assess_credit_profile` (and `assess_risk_trend` if a trend question is asked)
- **Policy/coaching question** → PDF-grounded RAG (`retrieve_pdf_chunks` +
  `check_sufficiency` from `kickoff_prompt.md`)
- **Off-topic** → escalate to human, per `kickoff_prompt.md`'s guardrails

This routing decision is genuinely new (production FinBuddy's own `/api/v1/intent`
endpoint classifies onboarding intents, a different category set — worth citing as
prior art for the *pattern*, but don't force-fit its output here). Build the Intent
Router as its own small classical classifier (e.g. TF-IDF/embedding + Logistic
Regression, or start with keyword rules and graduate to a trained classifier once
you have labeled routing examples) — and this is the **only** model in this project
that needs its own full MLOps loop, because it's the only model this project
actually creates.

## MLOps — scoped to what this project actually owns

Do not duplicate production FinBuddy's MLOps for the four models served by
`/api/v1/score` — that governance already exists and re-implementing it here would
be redundant, not additive. This project owns MLOps for exactly two things:

**1. The Intent Router** (fully owned — build the whole loop):

```
Data & Feature   — versioned (message, route_label) dataset, labeled from real
                    or realistic example queries across all three routes
    ↓
Train & Track    — log every run to MLflow; if you compare approaches (e.g.
                    keyword rules vs TF-IDF+LogReg vs embedding+LogReg), log all of them
    ↓
Model Registry   — staging → human sign-off → production; archive old versions
    ↓
Serve            — inline in the LangGraph router node
    ↓
Monitor          — track routing-confidence distribution and misroute rate (from
                    user corrections or eval-harness scenarios) over time
    ↓
Retrain Trigger  — accuracy decay or a misroute-rate breach queues retraining on
                    the freshest labeled routing log
```

**2. The Risk-Trend Classifier tool** (partial ownership — you execute it, but you
didn't train it): monitor its **input distribution** for drift, using two DIFFERENT
checks for two different jobs — found necessary by actually wiring this up, not
assumed upfront. A **per-request** check (is this one incoming value plausible?)
cannot use PSI the way `drift_report.py` does: PSI is a batch-comparison statistic,
and verified directly that applying it to a single value (n=1) against a 10-bin
reference always reports "drift" regardless of whether the value is typical — one
point concentrating 100% of its mass in one bin against a reference spread ~10%/bin
is guaranteed to diverge, which is a property of the metric, not a signal about the
data. `check_input_drift()` uses a percentile-band (1st/99th) out-of-range check for
this instead. The batch-PSI machinery is still implemented (`fit_reference_bins()`),
for a future weekly-dashboard consumer, matching the drift-ladder design below, but
is not called per-request. Retraining the model, if ever needed, is production
FinBuddy's responsibility, not this project's — this project only needs to **detect
and flag**, then fall back to the credit score's own `is_anomalous` signal from
`assess_credit_profile` if the risk-trend inputs look unreliable.

### Drift response ladder (for the Intent Router and for Risk-Trend input monitoring)

| Tier | Trigger | Action |
|---|---|---|
| GREEN · Monitor | Router misroute rate low, PSI < 0.10 on risk-trend inputs | No action; weekly check |
| AMBER · Investigate | Misroute rate rising, or PSI 0.10–0.20 | Review routing errors; for risk-trend, flag results as lower-confidence in the response |
| RED · Auto-Retrain Flag (router only) | Misroute rate breaches threshold | Retrain the Intent Router on the freshest labeled log |
| GOVERNANCE GATE · Promote (router only) | — | New router version must beat the current one on the eval harness and get human sign-off before serving |

## Governance gates (scoped to this project)

| Gate | Owner | Pass criteria |
|---|---|---|
| Technical review | You (Data Science role) | Intent Router accuracy on the eval harness meets a stated target; `assess_credit_profile` / `assess_risk_trend` tool-call error handling verified (see Guardrails in `kickoff_prompt.md`) |
| Fairness | Inherited, not re-audited | The fairness-mitigated decision comes from production FinBuddy's own F-012 audit — cite it, don't recompute it. If you introduce any NEW decision surface (e.g. the Intent Router treating some phrasing differently by language), audit *that* with Fairlearn — that would be a genuinely new fairness question, not the one production already answered |
| Business/regulatory sign-off | You (PM + "Legal" role) | Grounding gate and PII-handling guardrails verified working; this project never stores or forwards real UPI signals anywhere except the production scoring API call itself |
| Production approval | You (MLOps role) | Intent Router champion/challenger canary passes; risk-trend input-drift monitor green at time of promotion |

## Updated architecture (adds a seventh layer to `kickoff_prompt.md`'s six)

```
1. API Layer         - FastAPI endpoint, auth, request validation, streaming
2. Agent Layer        - ReAct loop: Intent Router decides the path, then plans
                        tool calls (RAG tools, or the two credit-assessment tools)
3. Tool Layer         - retrieve_pdf_chunks, check_sufficiency, source-metadata,
                        assess_credit_profile (calls live scoring API),
                        assess_risk_trend (loads artifact read-only)
4. Memory Layer       - short-term (LangGraph state); long-term deferred, named as such
5. Guardrail Layer    - input validation, tool policy (esp. never forwarding raw
                        UPI signals anywhere but the trusted scoring API), output
                        validation
6. Observability      - tracing, evaluation hooks, cost/step logging
7. MLOps Layer (NEW)  - full loop for the Intent Router only; drift-monitoring
                        (not retraining) for the Risk-Trend tool's inputs; explicit
                        non-duplication of production's F-001/003/006/012 governance
```

## Demo narrative addition

`kickoff_prompt.md` already specifies STAR structure and a resume-line template.
This project's own featured deep-dive is now **correctly** the Risk-Trend tool
integration, told honestly: "we didn't retrain this — we consumed FinBuddy's
already-proven Ridge-vs-Lasso, multi-split-evaluated classifier as a read-only tool,
and built the *new* MLOps loop (train/track/registry/drift/retrain) around the one
model this project actually created — the Intent Router." That distinction (what's
reused-and-governed-elsewhere vs what's new-and-owned-here) is a stronger, more
honest story than claiming to have built five ML models from scratch, and it directly
answers the "why didn't you just retrain everything" question before it's asked.

## Updated milestone checklist (supersedes `kickoff_prompt.md`'s "First milestone" section)

1. Scaffold the repo per the seven-layer structure above.
2. Collect the real RBI/DPDP PDFs for the demo corpus; ingest via Docling into Chroma.
3. Build `assess_credit_profile` as a thin, guarded HTTP client against production
   FinBuddy's live `/api/v1/score` endpoint (handle timeouts/errors explicitly —
   see Guardrails in `kickoff_prompt.md`).
4. Build `assess_risk_trend` by loading `risk_trend_logreg.joblib` read-only and
   wrapping it as a tool with the same input contract as `train_risk_trend.py`
   expects (the delta-feature columns) — verify against a few known synthetic
   examples before trusting it in the graph.
5. Build the Intent Router: start with a labeled example set across the three
   routes, train a simple classifier, log the run to MLflow, register it.
6. Wire the Intent Router + all four tools (2 RAG, 2 credit-assessment) into the
   LangGraph state machine from `kickoff_prompt.md`, with the Generate node still
   the only LLM call.
7. Add the Intent Router's full MLOps loop (registry states, drift monitor on
   misroute rate) and the Risk-Trend input-drift monitor (PSI, detect-only).
8. Wire the evaluation harness, guardrail tests (including "credit-assessment tool
   times out / production API is down" as an explicit tested failure case),
   tracing, and CI gate from `kickoff_prompt.md`; verify the local-Ollama fallback
   once and keep the proof.
9. Containerize and deploy to a **new** HuggingFace Space; confirm the existing
   production FinBuddy Spaces are untouched.
