---
title: FinBuddy Agentic RAG (LangGraph)
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# FinBuddy Agentic RAG (LangGraph)

**Repo:** [github.com/bhavyabhandary-star/finbuddy-agentic-rag-langgraph](https://github.com/bhavyabhandary-star/finbuddy-agentic-rag-langgraph)
**Live demo:** [huggingface.co/spaces/BhavyaBhandary/finbuddy-langgraph-agent](https://huggingface.co/spaces/BhavyaBhandary/finbuddy-langgraph-agent)

A local-first, agentic RAG system that answers FinBuddy policy/coaching questions
grounded in real RBI/DPDP PDFs, and orchestrates FinBuddy's existing production
credit-scoring pipeline as tools — real Setu AA Feed sandbox data in, the live
scoring API + Risk-Trend model for inference — built with LangChain + LangGraph.

## Current state (2026-11)

| | |
|---|---|
| **Live** | HF Space snapshot built from `9d41b3a` (2026-11-05), running the promoted Intent Router `20261027T083951`. No Setu credentials there: an `aa_handle` request fails safe to a human hand-off |
| **Router accuracy on unseen queries** | **~80%** (blind set, n=30, roughly ±14 points); 97% on the dev set it was hardened against (optimistic). The old "100% on the eval harness" was a train-on-test leak and is retired. |
| **When the router is wrong** | It fails safe: off-topic and insufficient retrieval both end in a human hand-off; a credit misroute with no signals escalates with no score |
| **Failure behaviour** | LLM outage, timeout, invalid JSON, or an answer stating a figure not in the sources → fixed human-escalating answer with a `failure_reason`; never a 500 |
| **Privacy / access** | PAN, Aadhaar and mobile numbers redacted before routing, the LLM and the logs; optional `X-API-Key`; per-client rate limit (30/min) |
| **Tests** | 111 pass, 3 skip (the real-Claude tests, no billing) |
| **Still open** | Hindi output garbled (accepted), Kannada fixed fallback, demo API open unless a key is set, caller identity unverified, scoring models trained on synthetic data, no console feedback buttons, RAGAS blocked on the dev machine |

Details, before/after measurements and the full list of what is *not* solved:
[`docs/robustness_checklist.md`](docs/robustness_checklist.md).

**Read the design docs first, in this order:**

1. [`kickoff_prompt.md`](kickoff_prompt.md) — role, constraints, six-layer
   architecture, the LangGraph state machine, guardrails, cost/latency, evaluation,
   deployment.
2. [`build_prompt.md`](build_prompt.md) — resolved decisions, the ML tools this
   agent orchestrates (each mapped to its real production purpose, not repurposed),
   the MLOps this project actually owns, and the milestone checklist this scaffold
   follows.
3. [`docs/langgraph_state_machine.md`](docs/langgraph_state_machine.md) — the
   state machine's actual nodes/edges, as a Mermaid diagram generated directly
   from the compiled graph (not hand-drawn).

This is a **separate project** from `finbuddy-project` (the production FinBuddy
build). It never modifies that repo — it consumes its live scoring API and one
trained artifact, read-only.

## Layout (seven layers, per `build_prompt.md`)

```
api/            FastAPI endpoint, request validation, streaming            (Layer 1)
                + security.py (API key, rate limit), response_cache.py,
                feedback.py (verdict capture for the retrain loop)
agent/          LangGraph state machine: Intent Router, planning, state    (Layer 2)
tools/          retrieve_pdf_chunks, check_sufficiency, assess_credit_profile,
                assess_risk_trend                                          (Layer 3)
memory/         short-term (LangGraph state); long-term explicitly deferred (Layer 4)
guardrails/     input validation + PII redaction, tool policy, output
                validation, runtime grounding check                        (Layer 5)
observability/  tracing (PII-redacted), cost/token/latency accounting,
                live /metrics                                              (Layer 6)
mlops/          Intent Router's full train/track/registry/drift loop;
                Risk-Trend tool's input-drift monitor (detect-only)        (Layer 7)

ingestion/      PDF ingestion (Docling) → chunking → Chroma vector store;
                Setu AA Feed (real sandbox UPI data → 8 signals)
eval/           held-out + blind routing sets and the CI gate; independent-judge
                RAG eval (judge_eval.py); RAGAS (ragas_eval.py, see Status)
tests/          pytest suite
ui/             finbuddy_console.html -- persona-based credit-assessment +
                policy Q&A console, deployed via GitHub Pages (see Deployment)
docs/           additional design notes, the LangGraph diagram, demo checklist

config/         every tunable and env-driven setting in one place (settings.py):
                model params, thresholds, loop/retry caps, endpoints, eval gate.
                Secrets are read from the environment, never given in-code defaults
prompts/        every LLM-facing instruction, versioned (prompts/<name>/vN.txt);
                a published version is immutable -- change wording by adding vN+1
                and moving ACTIVE_VERSIONS. The live version is written into each
                generate-step trace
```

### How this maps to the standard AI-agent project layout

| Standard folder | Here | Notes |
|---|---|---|
| `config/` | `config/` | Added: tunables + env settings centralised; `tests/test_config.py` fails if a module re-defines one |
| `data/` | `data/`, `ingestion/`, `chroma_data/` | Raw PDFs and the cached Setu profile are gitignored by design |
| `agents/` + `workflows/` | `agent/` | `graph.py` is the workflow (LangGraph state machine), `nodes/` are the steps |
| `tools/` | `tools/` | Plus `guardrails/tool_policy.py` for safe-call/retry/confirmation rules |
| `memory/` | `memory/` | Short-term only; long-term memory deliberately deferred |
| `prompts/` | `prompts/` | Added: prompts moved out of `agent/nodes/generate.py`, versioned, hash-frozen in tests |
| `evaluation/` | `eval/` | Held-out + blind routing sets (leak-tested), independent-judge RAG eval, CI gate |
| `app/` | `api/`, `ui/` | FastAPI endpoint + console UI |
| `scripts/`, `docs/`, Docker | same names | |
| *(not in the standard list)* | `guardrails/`, `observability/`, `mlops/` | Kept: input/output/tool guardrails, tracing, Intent Router + drift monitors |

Directory names that differ from the standard list (`agent/`, `eval/`, `api/`) were
kept on purpose: the Dockerfile's `uvicorn api.main:app`, CI's `eval.evaluate_agent`
and every import depend on them, so renaming would be churn with deploy risk and no
behaviour benefit.

## Status

Milestone steps 1–6 (of `build_prompt.md`'s checklist) are real and verified end
to end, not stubs: PDF ingestion (5 real RBI/DPDP PDFs → Chroma), cross-encoder
re-ranking, a trained Intent Router (real MLOps loop), the credit-assessment
tools (live production API + real Risk-Trend artifact + real Setu AA sandbox
data), and the full LangGraph state machine, run for real end to end for all
three routes. See each module's docstring for what's real vs. not yet
implemented — this project follows the same "what's real vs. demo-grade"
honesty convention as `finbuddy-project`'s own README.

Milestone step 9 (deployment) is also done: containerized and deployed to a new,
separate HuggingFace Space at
[huggingface.co/spaces/BhavyaBhandary/finbuddy-langgraph-agent](https://huggingface.co/spaces/BhavyaBhandary/finbuddy-langgraph-agent),
verified live by calling `/agent/run` directly against the deployed instance and
confirming all three routes (grounded policy answer with citation, off-topic
refusal, insufficient-context escalation) work end to end in production — not
just locally.

> **Read the RAGAS paragraph below as history.** Its figures are from three questions,
> judged by the same Qwen model that wrote the answers, and RAGAS cannot be imported on the
> dev machine any more (a Windows Application Control policy blocks one of its native
> dependencies). For current RAG answer quality use `python -m eval.judge_eval` (nine
> questions, an independent Llama judge, escalations reported rather than averaged in) --
> results in `docs/judge_eval_results.json`.

The RAGAS evaluation (`eval/ragas_eval.py`) is also real now, not a stub: it
runs real retrieval + real generation against the ingested corpus, judged by
the same HF-hosted model this project uses for generation (no OpenAI
dependency). It initially surfaced a genuine retrieval-quality gap, not a clean
pass: one of three real queries (DPDP consent requirements) retrieved the
wrong chunk, because `fixed_size_chunks`'s 200-word sliding window sliced
across a real section boundary. Fixed by switching `ingestion/build_corpus.py`
to `semantic_chunks` (paragraph-boundary splitting, which Docling's markdown
export already aligns with each PDF's actual clause structure) and rebuilding
`chroma_data/` (379 → 1021 chunks). Verified for real, not assumed: the
previously-failing query now ranks the correct clause #1, and re-running RAGAS
confirms it — faithfulness 0.92 → 0.96, answer relevancy 0.67 → 0.81, context
precision 0.67 → **0.97**, context recall 0.67 → **1.00**. The fix is deployed
and confirmed live: the Space was redeployed with the rebuilt `chroma_data/`,
and `eval/ragas_eval.py --production` — which calls the deployed Space's real
`/agent/run` endpoint for the answer — reproduces the fix in production
(faithfulness 1.00, context precision 0.97, context recall 1.00; all three
answers correct, including the previously-wrong DPDP consent question). Full
before/after/production detail in `docs/ragas_eval_results.json`.

The console UI (`ui/finbuddy_console.html`) is also real, not a static
mockup: persona-based credit assessment (Simple/Advanced modes, editable UPI
signals), real SHAP-explained factor cards, an honest per-factor eligibility
roadmap (each step reuses the scoring tool's own real `action` tip — no
fabricated loan amounts or unlock timelines), and a policy-Q&A chat with a
trace panel showing the real route/grounding/citations behind each answer.
Deployed via GitHub Pages at
[bhavyabhandary-star.github.io/finbuddy-agentic-rag-langgraph/finbuddy_console.html](https://bhavyabhandary-star.github.io/finbuddy-agentic-rag-langgraph/finbuddy_console.html)
(auto-deploys on any push touching `ui/**`, see `.github/workflows/deploy-pages.yml`).
Note: the console cannot run inside a claude.ai Artifact preview — Artifacts'
CSP blocks `fetch()` to any host outside a small CDN allowlist, so
`/agent/run` calls there always fail with `Failed to fetch`; GitHub Pages has
no such restriction and is what the live link above actually uses.

CI's own `Evaluation gate (routing accuracy)` step had a real, previously
undiscovered bug, not a hypothetical one: `mlops/intent_router/registry_store/`
(the trained Intent Router's model) is deliberately gitignored — a real,
already signed-off model lives there on a developer's machine and gets
bundled into the HF Space deploy snapshot separately — so it never existed
in a fresh CI checkout. `classify_intent()` silently fell back to a crude
6-keyword heuristic instead, which always returns `confidence=0.5` and
happened to sit exactly at the eval gate's 75% pass floor by coincidence —
every CI log ever produced showed that flat 0.5 on all four scenarios,
confirmed by reproducing the identical behavior locally (moving
`registry_store/` aside reproduces CI's exact output). This meant the gate
could never have caught a real model regression. Fixed by having CI train
and promote a real model into that job's own disposable filesystem before
the gate runs (`mlops/intent_router/train_and_promote.py`) — this is
deliberately *not* a real production promotion, which still requires
genuine human sign-off per `registry.py`'s governance gate; it's discarded
when the runner ends. Also pinned `LogisticRegression`'s own `random_state`
(the `saga` solver has randomness independent of the train/test split's own
seed), since training wasn't fully reproducible run to run before that. The
gate now scores a real model rather than the keyword fallback.

**Correction (2026-10-25):** the "100% routing accuracy" this section used to
quote was itself a leak. The four scenarios in `eval/scenarios.py` are
word-for-word members of the router's training set, so the gate was measuring
memorisation. The gate now scores queries the router has never seen
(`eval/heldout_scenarios.py`, `eval/blind_scenarios.py`); the honest figure
is ~80% on the blind set. See `docs/robustness_checklist.md`.

## Robustness

An 8-point production-readiness audit (data leakage, evaluation, failure
handling, latency, cost, security, monitoring, feedback) drove a set of fixes;
`docs/robustness_checklist.md` has the before/after table, the measurements,
and -- explicitly -- what is still open. In short:

- **Failure handling:** a provider outage, timeout or invalid JSON becomes a fixed
  human-escalating answer (`failure_reason`), never a 500; a runtime grounding check
  (`guardrails/grounding.py`) refuses an English policy answer that states a figure
  not in the retrieved text.
- **Privacy / access:** PAN, Aadhaar and mobile numbers are redacted before routing,
  the LLM and the trace files; optional `X-API-Key`; per-client rate limit; validated
  `session_id` (it used to become a filename).
- **Cost / latency:** startup warm-up, a TTL cache for grounded policy answers, a
  non-blocking stream endpoint, per-run token and latency accounting (cost stays
  `null` until `LLM_PRICE_*` is set -- it is not guessed).
- **Evaluation:** held-out and blind routing sets with a leak test; RAG answers graded by an
  independent judge (`eval/judge_eval.py`; RAGAS itself is blocked by an Application Control
  policy on the dev machine).
- **Monitoring / feedback:** `GET /metrics`; router drift from user feedback and from a
  label-free confidence signal; `POST /agent/feedback` -> reviewed corrections -> retrain.

### API surface

| Endpoint | Purpose | Protected by key + rate limit |
|---|---|---|
| `GET /health` | liveness probe | no |
| `POST /agent/run` | the agent; returns `answer`, `sources`, `route`, `failure_reason`, `guardrail_flags`, `pii_redacted`, `cache_hit`, `run_id`, ... | yes |
| `POST /agent/run/stream` | same run as server-sent events (step names, then the answer) | yes |
| `POST /agent/feedback` | a verdict on a `run_id` (`rating`, optional `correct_route`, `comment`) -> `feedback/feedback.jsonl`, `reviewed: false` | yes |
| `GET /metrics` | volume, escalation rate, fixed-fallback reasons, p50/p95, cache hit rate, tokens, router drift | key only |

"Protected" means: `X-API-Key` is enforced **only if** `FINBUDDY_API_KEY` is set (the public demo
leaves it unset because the GitHub Pages console cannot keep a secret); the rate limit is always on.

### The retrain -> gate -> promote loop

1. Users' verdicts land in `feedback/feedback.jsonl` (gitignored, PII-redacted) with `reviewed: false`.
2. A human reviews them and sets `reviewed: true`; only those become training data
   (`mlops.intent_router.data.load_feedback_examples`).
3. `python -m mlops.intent_router.train` writes a candidate; `INTENT_ROUTER_ARTIFACT=<path>`
   evaluates or runs it **without** promoting it.
4. It must beat the current production router on `eval/heldout_scenarios.py` and
   `eval/blind_scenarios.py` (`python -m eval.evaluate_agent --gate`), and the held-out sets must
   never overlap the training data (`tests/test_eval_integrity.py` fails the build if they do).
5. A named human signs it off via `registry.promote_to_production(...)`.
   `train_and_promote.py` is CI-only and labels itself as such.
6. Redeploy: the deploy snapshot bundles the production registry, so a promotion only reaches
   the Space after `scripts/prepare_hf_space_deploy.sh` + a push.

The last promotion is documented in `docs/robustness_checklist.md` (router `20261027T083951`,
blind set 73.3% -> 80.0%, off-topic recall 66% -> 91%).

## Real-data readiness

The credit models this agent calls (F-001 score, F-003 SHAP explainer, F-006 anomaly, Risk-Trend) were trained on
**synthetic** data, and production FinBuddy's own generator says they must be retrained on real, consented Setu AA data
before any real lending decision relies on them. That has **not** happened, and `mlops/real_data/` exists so the repo can say
exactly why, and refuse to pretend otherwise:

```bash
python -m mlops.real_data.readiness   # per-model verdict: BLOCKED / ELIGIBLE, and what is missing
python -m mlops.real_data.shift       # where real profiles fall vs the synthetic training data
python -m mlops.real_data.collect --vua <mobile>   # pull one consented profile + write the consent register
```

Today: 1 profile, and it is Setu **sandbox** (mock accounts), with 0 repayment outcomes, so all four models are BLOCKED.
That profile is outside the synthetic training range on 3 of 8 signals (its average transaction is about 14x the
synthetic 99th percentile). Full status, thresholds and the path to real data: [`docs/real_data_readiness.md`](docs/real_data_readiness.md).

## Live Account Aggregator flow

`POST /agent/run` accepts an `aa_handle` (a 10-digit mobile, or `<mobile>@<AA handle>` such as `9999999999@finvu`; Setu's sandbox no longer accepts `@onemoney`). For a credit question the
graph then runs **consent, pause, fetch, normalise, score** instead of reading a cached profile:

1. First call returns `consent_status: PENDING` and a `consent_url`. The person opens it and approves; nothing is fetched before that.
2. Call again with the same `session_id`. If the consent is now ACTIVE the data is fetched, reduced to the 8 signals, and scored by the
   existing credit path. Declined, expired or unavailable ends in a hand-off with no score.

Raw bank data is never put in graph state, traces, or an LLM prompt. The public demo has no `SETU_*` credentials, so there this path
fails safe with a hand-off; to run it for real you need your own Setu sandbox credentials in `.env` and a human to approve the
consent. Optional `POST /webhooks/setu/consent` (enabled only when `SETU_WEBHOOK_SECRET` is set) lets Setu notify instead of waiting
for the next request. Sandbox data is Setu's mock bank accounts: it proves the flow, not borrower behaviour (see Real-data readiness).
Design and tests: `docs/langgraph_state_machine.md`, `tests/test_aa_flow.py`.

## Language support

`response_language` accepts `en`, `hi`, and `kn` (`agent/nodes/generate.py`'s
`LANGUAGE_NAMES`) — but the three are handled very differently, based on real
testing against the deployed Space's provider, not assumption:

- **English (`en`)** — real generation, no known issues.
- **Hindi (`hi`)** — real generation. A guardrail
  (`guardrails/output_guardrails.py::is_expected_script`) retries once if the
  model answers in the wrong script entirely, then falls back to a fixed
  message if the retry also fails. Real production testing still occasionally
  turns up smaller leaked-token noise (a stray foreign word or character)
  inside an otherwise-correct-script answer — the wrong-script guardrail can't
  catch that, and it's an accepted, known limitation rather than a bug being
  chased further.
- **Kannada (`kn`)** — generation is skipped unconditionally. `generate_node`
  never calls the LLM provider for a `kn` request; it always returns a fixed,
  human-authored fallback answer telling the user to ask in English instead.
  This is stronger than Hindi's handling because real testing showed Kannada
  output broken at a deeper level: even when the model happened to land in
  correct Kannada Unicode script, the content itself was incoherent gibberish,
  not real Kannada — no retry can fix that. Kept UI-disabled ("coming soon")
  in `ui/finbuddy_console.html` for the same reason. The fixed fallback (here and Hindi's)
  returns no `sources`: it was not generated from the retrieved text, so it must not cite it.
- **Tamil and Telugu are out of scope** — not in `LANGUAGE_NAMES`, no UI entry.
  An early real test of both surfaced the same wrong-script/gibberish failures,
  more severely (including stray CJK characters).

This is a model-capability limitation, not a prompt bug: the system prompt
template in `agent/nodes/generate.py` was checked directly and contains
nothing that could cause these leaks. It stems from the deployed Space
defaulting to `LLM_PROVIDER=huggingface` (a free-tier ~7B open model) rather
than `claude`, since Claude API billing isn't set up for this deployment —
see the provider setup below.

## Running it (once dependencies are installed)

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload --port 8010
```

**Choose an LLM provider** in `.env` via `LLM_PROVIDER` (`claude` / `huggingface`
/ `ollama`) and fill in the matching credential. Claude needs `ANTHROPIC_API_KEY`
*and* a funded account — a `claude.ai` chat subscription does **not** cover API
usage, they're billed separately. `huggingface` needs `HF_TOKEN` (a free HF
account's token) and works out of the box with the default model/provider in
`.env.example`. `ollama` needs a local Ollama install.

**One-time setup for the Risk-Trend tool** (`assess_risk_trend`): copy the real,
already-trained artifact from `finbuddy-project` — this project consumes it
read-only and never regenerates it itself:

```bash
cp "<path-to-finbuddy-project>/scoring_service/models/artifacts/risk_trend_logreg.joblib" \
   mlops/risk_trend_monitor/artifacts/risk_trend_logreg.joblib
python -m mlops.risk_trend_monitor.build_reference_distribution \
   "<path-to-finbuddy-project>/scoring_service/data/synthetic_risk_trend_dataset.csv"
```

**One-time setup for the Setu AA Feed** (`ingestion/setu_feed.py`): copy the
real, already-pulled sandbox profile — same read-only pattern:

```bash
cp "<path-to-finbuddy-project>/scoring_service/data/setu_real_profiles.jsonl" \
   data/setu_real_profiles.jsonl
```

Pulling a *fresh* profile (`ingestion.setu_feed.pull_fresh_profile()`) needs
real `SETU_*` credentials in `.env` and a human to approve the consent URL in a
browser — there is no headless way to do this (Setu's own sandbox protocol
requires it). Not needed to use the already-pulled profile above.

The tests that exercise these real dependencies skip automatically if the
artifact/profile/token isn't present (same pattern throughout — check each
test file's `pytest.mark.skipif` for exactly what it needs).

```bash
pytest tests/ -v                      # 111 pass, 3 skip when no Claude billing
python -m eval.evaluate_agent --gate  # held-out + blind routing sets, per-route recall
python -m eval.judge_eval             # RAG answers graded by an independent (Llama) judge
```

Optional environment (all documented in `.env.example`): `FINBUDDY_API_KEY` (turn on key
auth), `TRUST_PROXY_HEADERS` (automatic on HF Spaces), `LLM_PRICE_INPUT_PER_1K` /
`LLM_PRICE_OUTPUT_PER_1K` (cost is reported as unknown until set), `RAGAS_JUDGE_MODEL`,
`INTENT_ROUTER_ARTIFACT` (run a candidate router without promoting it), `SKIP_WARMUP=1`
(tests). On Windows, run the project venv from PowerShell: some native DLLs
(scikit-learn, tiktoken) are blocked by Application Control when launched from other shells.

## Deployment

Two independent deployment targets:

- **Backend API** — HuggingFace Spaces, a **new, separate Space** from the
  existing production FinBuddy Spaces. See `Dockerfile` and
  `scripts/prepare_hf_space_deploy.sh`, which builds a single-commit snapshot
  including the gitignored real artifacts the dev repo doesn't track
  (the trained Intent Router model, `chroma_data/`, the Risk-Trend artifact,
  the Setu profile) — it never touches your HF credentials, so the printed
  `git push` command is yours to run. It deliberately leaves out `feedback/` (user text)
  and `docs/*.pptx|*.docx` (binaries HF rejects without LFS). **Promote the router first**:
  the snapshot carries whatever is in `mlops/intent_router/registry_store/production/`.
  Behind HF's proxy the rate limiter keys on `X-Forwarded-For` (automatic when `SPACE_ID`
  is set); a Space's variables (`LLM_PROVIDER`, `HF_TOKEN`, optionally `FINBUDDY_API_KEY`)
  are set in its Settings, never in the snapshot.
- **Console UI** — GitHub Pages, auto-deployed by
  `.github/workflows/deploy-pages.yml` on any push touching `ui/**`. See the
  Status section above for why this is a separate deployment target from the
  HF Space rather than the same static file being embedded there.
