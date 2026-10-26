# Robustness checklist: what was audited, what changed, what is still open

Audit date: 2026-10-25. Scope: this repo (the agentic RAG layer). The production
`finbuddy-project` repo was read, not changed -- its open items are listed at the end.

Every claim below is backed by a test (`tests/test_robustness_units.py`,
`tests/test_api_robustness.py`, `tests/test_eval_integrity.py`) or by a measurement named
in the "Evidence" column. "Open" means exactly that -- not done.

| # | Check | Before the audit | Now | Still open |
|---|---|---|---|---|
| 1 | Data quality and leakage | All 4 eval scenarios were word-for-word in the router's training set, so "100% on the eval harness" and CI's gate measured memorisation. The shipped router was also fit on only 50% of the data (the split it was selected on). | Held-out sets the router never saw (`eval/heldout_scenarios.py`, `eval/blind_scenarios.py`); `test_eval_integrity.py` fails on any exact or near-duplicate (Jaccard >= 0.7) overlap with training or reviewed-feedback data. The winning config is refit on all labelled data. | Labels were written by the router's author; a second reviewer should audit them. Scoring-model data in the production repo is synthetic (see end). |
| 2 | Evaluation beyond accuracy | Routing only, n=4, in-training. RAGAS n=3, judged by the same Qwen model that generated the answers. | Gate scores held-out queries, fails on per-route recall < 60% and on either set < 75%; reports per-route precision/recall, a confusion list, accuracy per edge-case tag (typo, Hinglish, adversarial, PII, ambiguous) and every miss. RAG quality: 9 questions graded by an independent judge (Llama-3.1-8B, a different family from the Qwen generator) via `eval/judge_eval.py`; escalations are reported, not averaged in or hidden. **RAGAS itself could not be re-run**: its native dependency (tiktoken) is blocked on this machine by a Windows Application Control policy, so `eval/judge_eval.py` is a labelled substitute, not RAGAS. | n is small (66 + 30 routing queries, 9 RAG questions) and an 8B judge is a noisy grader. No fairness slices for the agent. No eval of credit-explanation faithfulness. The 2026-09 RAGAS numbers in `docs/ragas_eval_results.json` (n=3, self-judged) should not be quoted as current. |
| 3 | Hallucinations and failure handling | Provider outage / timeout / bad JSON surfaced as an HTTP 500. No runtime check that an answer is supported by the retrieved text. | Any provider failure or twice-invalid JSON becomes a fixed, human-escalating answer (`failure_reason` says why). `guardrails/grounding.py` rejects an English policy answer containing a figure absent from the retrieved text, or with < 50% content-word overlap. Fixed Kannada/Hindi fallbacks no longer cite sources they were not generated from. Any unexpected exception in a run also fails closed. | Grounding is lexical: it can reject a correct, heavily paraphrased answer (a false escalation) and does not catch a wrong claim that reuses context words. Not applied to Hindi/Kannada. Measured: 1 of 36 real generations (3%) was rejected as ungrounded (the other 35 passed; lowest word overlap 0.73) -- I did not capture that one answer, so I cannot say whether the rejection was right. The hi/kn "LLM unavailable" strings are unreviewed by a native speaker. |
| 4 | Latency and scalability | Cross-encoder + index loaded inside the first request (~20 s measured on the Space). `/agent/run/stream` blocked the event loop for the whole run. No LLM timeout. | Warm-up at startup (`SKIP_WARMUP=1` to disable); stream consumed through a thread pool (test: `/health` answers in ms while a slow run streams; live: 4-27 ms during a real streamed run); `LLM_TIMEOUT_SECONDS=45` on every provider; one retry for a transient provider error (1 of 4 back-to-back real calls failed once, which would otherwise have escalated an answerable question); per-client rate limit (live: 40 rapid calls -> 30 served, #31 got 429 with Retry-After). | No load test beyond a handful of concurrent calls. Single process; the rate limiter and cache are per-process. |
| 5 | Inference cost | No cost visibility (`RunCostTracker` existed but nothing called it; its prices were Claude's, the model is Qwen). No caching. | Every LLM call records tokens (provider-reported when available, else a labelled estimate), latency and -- only if `LLM_PRICE_*` is set -- cost; written to each trace and `/metrics`. TTL/LRU cache for grounded policy answers (measured 0.03 s vs ~5 s cold). Classical-first routing already avoids the LLM for off-topic and Kannada. | Cost is **unknown** until a price is configured -- not guessed. No quantization / smaller-model trial was run, so no saving is claimed. |
| 6 | Security and access control | No auth, no rate limit, open CORS, raw queries (with any PAN/phone) sent to a third-party LLM and written to trace files; client `session_id` became a trace filename (path traversal); the injection-marker check was a no-op. | Opt-in `X-API-Key` (constant-time compare) on `/agent/*` and `/metrics`; sliding-window rate limit; PAN/Aadhaar/mobile redacted before routing, LLM and logging, and again inside the tracer (live: 0 of 95 trace files contain a test PAN/phone); the retrievers get the question without its PII sentence -- measured, a PII preamble dropped a well-formed question's cross-encoder score from 0.987 to 0.071 and redaction alone only to 0.036; `session_id` pattern-validated and trace filenames sanitised; query length cap; supported-language whitelist; injection markers in retrieved text are neutralised and flagged. | The API still cannot tell *who* the caller is, so `credit_signals` are caller-supplied and `use_setu_feed` serves the cached sandbox profile to any caller (key-gated only if `FINBUDDY_API_KEY` is set). The deployed demo stays open unless the key is set. CORS remains `*`. The PII patterns are India-specific and conservative, not exhaustive. |
| 7 | Monitoring and model drift | `record_routing_outcome()` was never called, so the misroute rate was always 0 and the tier always green. Traces were files on an ephemeral disk with no aggregate view. | `GET /metrics`: volume, escalation rate, fixed-fallback reasons, p50/p95, cache hit rate, tokens, router drift. Drift has two signals: misroute rate from human feedback, and a label-free routing-confidence-drift tier (baseline measured on held-out: 15.6% of queries below 0.40 confidence; amber at 30%, red at 45%). Neither reports a tier before enough samples. | No alerting, no persistence, per-process only. Thresholds are initial (no live traffic exists to calibrate them). Router confidence is weakly informative (mean 0.51 when right vs 0.43 when wrong), so the confidence signal is an early hint, not proof of drift. |
| 8 | Feedback and continuous improvement | No way to capture a user's verdict; nothing connected usage to retraining. | `POST /agent/feedback` (only for a run this process served; PII-redacted) -> `feedback/feedback.jsonl` -> a human sets `reviewed: true` -> `load_feedback_examples()` -> retrain. Corrections feed the misroute-rate signal immediately. | The console UI has no thumbs buttons yet (API only). Review is manual. Feedback is in-memory-indexed, so verdicts on pre-restart runs are refused. The retrained router still needs the held-out gate **and a human sign-off** to be promoted. |

## RAG answer quality (independent judge, n=9) -- `eval/judge_eval.py`, 2026-10-25

| Measure | Result | Type |
|---|---|---|
| Answered / escalated | 8 / 1 (the escalation was the runtime grounding check) | |
| Claim faithfulness (judge: Llama-3.1-8B) | 0.969 | LLM judge |
| Correct vs ground truth | 8 / 8 answered | LLM judge |
| Context recall | 0.97 | deterministic (word overlap) |
| Runtime grounding pass rate | 100% of answered | deterministic |

One judge-flagged unsupported claim (a DLG answer that paraphrased the cap loosely). Per-question rows
are in `docs/judge_eval_results.json`. Not comparable to the earlier RAGAS figures (different judge, more
questions, different metrics).

## Router: what the numbers actually are

| Measurement | Result | How to read it |
|---|---|---|
| Old harness (4 scenarios in the training set) | 100% | Memorisation. Never an accuracy claim. |
| First honest measurement, dev set, before any fix | 53/66 = 80.3% | The real starting point. Weak spots: finance-adjacent off-topic (investing, tax, crypto prices), injection-style text, paraphrased borrowing questions. |
| After hardening examples, **dev set** | 64/66 = 97.0% | Optimistic by construction: the examples were written to fix that set's failures. |
| After hardening, **blind set** (written afterwards, scored once) | 24/30 = 80.0% | **The number to quote.** Unseen-query routing accuracy is about 80%. |
| Per route on the 96 held-out queries | policy 94% / credit 91% / off-topic 91% recall | Misroutes mostly fail safe: off-topic and an insufficient retrieval both end in a human hand-off. |

## Router promotion

The retrained router is a **candidate**. It is evaluated through `INTENT_ROUTER_ARTIFACT` and
has not been promoted: `mlops/intent_router/registry.py` requires a named human sign-off, and
that stays a human decision. To promote after review:

```
python -m mlops.intent_router.train_and_promote   # CI-style ephemeral promotion only
# or, for a real promotion, stage the artifact and call promote_to_production(dir, signed_off_by="<your name>")
```

## What this repo did not change (production `finbuddy-project`)

Read-only here. Findings for that repo: scoring data is synthetic and its labels are a function
of the same features (reported AUCs describe the generator); the risk-trend generator adds label
noise so AUC lands near the target; scoring and RAG services are open by design (`CORS *`,
documented); no live traffic exists, so canary and drift reports run on held-out / synthetic data.
