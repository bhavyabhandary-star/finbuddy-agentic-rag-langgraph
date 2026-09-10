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
