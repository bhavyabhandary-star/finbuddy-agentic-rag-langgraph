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

**Read the design docs first, in this order:**

1. [`kickoff_prompt.md`](kickoff_prompt.md) — role, constraints, six-layer
   architecture, the LangGraph state machine, guardrails, cost/latency, evaluation,
   deployment.
2. [`build_prompt.md`](build_prompt.md) — resolved decisions, the ML tools this
   agent orchestrates (each mapped to its real production purpose, not repurposed),
   the MLOps this project actually owns, and the milestone checklist this scaffold
   follows.

This is a **separate project** from `finbuddy-project` (the production FinBuddy
build). It never modifies that repo — it consumes its live scoring API and one
trained artifact, read-only.

## Layout (seven layers, per `build_prompt.md`)

```
api/            FastAPI endpoint, request validation, streaming            (Layer 1)
agent/          LangGraph state machine: Intent Router, planning, state    (Layer 2)
tools/          retrieve_pdf_chunks, check_sufficiency, assess_credit_profile,
                assess_risk_trend                                          (Layer 3)
memory/         short-term (LangGraph state); long-term explicitly deferred (Layer 4)
guardrails/     input validation, tool policy, output validation           (Layer 5)
observability/  tracing, evaluation hooks, cost/step logging               (Layer 6)
mlops/          Intent Router's full train/track/registry/drift loop;
                Risk-Trend tool's input-drift monitor (detect-only)        (Layer 7)

ingestion/      PDF ingestion (Docling) → chunking → Chroma vector store;
                Setu AA Feed (real sandbox UPI data → 8 signals)
eval/           scenario harness + RAGAS metrics; CI gate
tests/          pytest suite
docs/           additional design notes as the project grows
```

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
  in `ui/finbuddy_console.html` for the same reason.
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
pytest tests/ -v
python -m eval.evaluate_agent --gate
```

## Deployment

HuggingFace Spaces — a **new, separate Space** from the existing production FinBuddy
Spaces. See `Dockerfile`.
