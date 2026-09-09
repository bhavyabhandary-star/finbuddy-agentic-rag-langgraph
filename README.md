# FinBuddy Agentic RAG (LangGraph)

A local-first, agentic RAG system that answers FinBuddy policy/coaching questions
grounded in real RBI/DPDP PDFs, and orchestrates FinBuddy's existing production
credit-scoring models as tools — built with LangChain + LangGraph.

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

ingestion/      PDF ingestion (Docling) → chunking → Chroma vector store
eval/           scenario harness + RAGAS metrics; CI gate
tests/          pytest suite
docs/           additional design notes as the project grows
```

## Status

Scaffold stage — module skeletons and interfaces are in place per the milestone
checklist in `build_prompt.md`; most functions are stubs with clear TODOs. See each
module's docstring for what's real vs. not yet implemented — this project follows
the same "what's real vs. demo-grade" honesty convention as `finbuddy-project`'s own
README.

## Running it (once dependencies are installed)

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY at minimum
uvicorn api.main:app --reload --port 8010
```

**One-time setup for the Risk-Trend tool** (`assess_risk_trend`): copy the real,
already-trained artifact from `finbuddy-project` — this project consumes it
read-only and never regenerates it itself:

```bash
cp "<path-to-finbuddy-project>/scoring_service/models/artifacts/risk_trend_logreg.joblib" \
   mlops/risk_trend_monitor/artifacts/risk_trend_logreg.joblib
python -m mlops.risk_trend_monitor.build_reference_distribution \
   "<path-to-finbuddy-project>/scoring_service/data/synthetic_risk_trend_dataset.csv"
```

The tests that exercise this tool skip automatically if the artifact isn't
present (same pattern as the API-key-gated integration tests).

```bash
pytest tests/ -v
python -m eval.evaluate_agent --gate
```

## Deployment

HuggingFace Spaces — a **new, separate Space** from the existing production FinBuddy
Spaces. See `Dockerfile`.
