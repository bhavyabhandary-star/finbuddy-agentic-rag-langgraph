"""The one place for tunables, model parameters and env-driven settings.

Before this module existed the same kinds of values were scattered across
generate.py, rag_tools.py, credit_tools.py, tool_policy.py, output_guardrails.py,
state.py, tracing.py and evaluate_agent.py. They now live here and those modules
import them (re-exporting the same names, so existing imports keep working).

Two kinds of values, deliberately kept apart:

1. Constants -- fixed, reviewed in code review like any other change.
2. Env-driven values -- API keys and deployment-specific endpoints. Secrets are
   NEVER given a default and never live in code; they are read from the
   environment (see .env.example for every variable).

Env is read at the same moments it always was, so behaviour is unchanged:
- SCORING_API_BASE_URL / RISK_TREND_ARTIFACT_PATH: at import time (as before in
  tools/credit_tools.py).
- LLM provider / model / HF token: at call time through the getters below (as
  before in generate.py and api/main.py), so tests can monkeypatch the env.
This module does not call load_dotenv(); api/main.py still does that.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- LLM / generation -------------------------------------------------------
DEFAULT_LLM_PROVIDER = "claude"  # claude | huggingface | ollama (env: LLM_PROVIDER)
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # ungated, broadly available across HF Inference Providers
DEFAULT_HF_PROVIDER = "featherless-ai"  # verified working; provider="auto" failed on this account
DEFAULT_OLLAMA_MODEL = "llama3.1"
# RAGAS judge: deliberately a DIFFERENT model family from the generator (Qwen). The first RAGAS
# runs used Qwen to grade Qwen's own answers, which flatters faithfulness; see eval/ragas_eval.py.
DEFAULT_RAGAS_JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_OUTPUT_TOKENS = 512  # explicit cap per kickoff_prompt.md's cost/latency section
# Hard ceiling on one LLM call. Without it a hung provider holds a worker thread forever;
# with it the call fails fast and generate_node escalates instead of hanging.
LLM_TIMEOUT_SECONDS = 45.0
# One retry for a transient provider failure. Measured: the hosted Qwen provider returned a
# one-off error in 1 of 4 back-to-back calls, which without a retry escalated a perfectly
# answerable question to a human.
MAX_LLM_CALL_RETRIES = 1
LLM_RETRY_BACKOFF_SECONDS = 1.0

# Languages generate_node may be asked to answer in. Tamil/Telugu are
# deliberately absent (out of scope, see README's "Language support").
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}

# ---- Retrieval --------------------------------------------------------------
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Calibrated against real (query, chunk) pairs: a true match scored 0.996,
# irrelevant chunks ~0.00003, so 0.5 sits with wide margin on both sides.
SUFFICIENCY_THRESHOLD = 0.5

# ---- Agent loop -------------------------------------------------------------
MAX_LOOP_ITERATIONS = 3  # hard cap per kickoff_prompt.md's cost/latency section

# ---- Guardrails -------------------------------------------------------------
MAX_RETRIES_PER_TOOL_CALL = 1
MAX_OUTPUT_VALIDATION_RETRIES = 1
# Runtime grounding check (guardrails/grounding.py): an English policy answer must be
# supported by the retrieved text. Any number in the answer that is not in the context
# or the question fails it outright; otherwise at least this share of the answer's
# content words must appear in the context. Calibrated against real answers -- see
# docs/robustness_checklist.md for how, and for the false-escalation trade-off.
GROUNDING_MIN_WORD_OVERLAP = 0.5

# ---- API hardening (api/security.py) -----------------------------------------
# Per-client sliding window. Generous on purpose: this stops a loop or a scraper from
# burning the LLM quota, it is not a billing tier.
RATE_LIMIT_MAX_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60
MAX_QUERY_CHARS = 1000  # reject absurdly long queries before they reach the router / LLM
MAX_FEEDBACK_COMMENT_CHARS = 500

# ---- Response cache (api/response_cache.py) ----------------------------------
# Policy answers come from a fixed corpus, so the same question has the same grounded
# answer. Only successful, non-escalated, non-credit answers are cached.
CACHE_TTL_SECONDS = 6 * 60 * 60
CACHE_MAX_ENTRIES = 256

# ---- Tools ------------------------------------------------------------------
SCORING_API_BASE_URL = os.environ.get(
    "FINBUDDY_SCORING_API_BASE_URL", "https://bhavyabhandary-finbuddy-scoring.hf.space"
)
RISK_TREND_ARTIFACT_PATH = os.environ.get(
    "RISK_TREND_ARTIFACT_PATH", "./mlops/risk_trend_monitor/artifacts/risk_trend_logreg.joblib"
)
# 60s, not tighter: the free-tier HF Space behind the scoring API sleeps when idle
# and took ~60-90s to cold-start on a real call (see tools/credit_tools.py).
SCORING_API_TIMEOUT_SECONDS = 60.0

# ---- Observability ----------------------------------------------------------
TRACE_DIR = Path("./traces")
FEEDBACK_PATH = Path("./feedback/feedback.jsonl")  # gitignored; see api/feedback.py
# Routing-confidence drift (observability/metrics.py + mlops/intent_router/drift_monitor.py):
# the share of live queries the router is unsure about. Baseline MEASURED on the 96 held-out
# queries: 15.6% score below 0.40. Amber is roughly 2x that baseline, red 3x. INITIAL values --
# held-out is not live traffic; recalibrate once there is real traffic.
LOW_ROUTE_CONFIDENCE = 0.40
LOW_CONFIDENCE_RATE_AMBER = 0.30
LOW_CONFIDENCE_RATE_RED = 0.45
DRIFT_MIN_SAMPLES = 30  # no confidence-drift tier is reported before this many routed requests
MISROUTE_MIN_FEEDBACK = 10  # ...nor a misroute tier before this many human feedback verdicts
LATENCY_WINDOW = 500  # most recent requests kept for p50 / p95

# ---- Evaluation -------------------------------------------------------------
# Below this fraction of scenarios routed correctly, `eval.evaluate_agent --gate`
# fails the build. (This is the only CI quality gate today.)
GATE_MIN_ROUTING_ACCURACY = 0.75
# The gate now scores eval/heldout_scenarios.py (queries the router never trained on),
# not the four in-training scenarios. It also fails if any one route's recall falls
# below this, because an accuracy average can hide a route that has collapsed.
GATE_MIN_PER_ROUTE_RECALL = 0.60


# ---- Call-time env getters (secrets: no defaults) ---------------------------
def llm_provider_name() -> str:
    return os.environ.get("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()


def hf_model() -> str:
    return os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)


def hf_provider() -> str:
    return os.environ.get("HF_PROVIDER", DEFAULT_HF_PROVIDER)


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def ragas_judge_model() -> str:
    return os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_RAGAS_JUDGE_MODEL)


def ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def api_key() -> str | None:
    """FINBUDDY_API_KEY. Unset = open demo API (as deployed today); set = every
    /agent/* and /metrics call must send it as X-API-Key. A secret, so no default."""
    return os.environ.get("FINBUDDY_API_KEY") or None


def trust_proxy_headers() -> bool:
    """Behind a reverse proxy the socket peer is the proxy, so without this every visitor
    shares ONE rate-limit bucket. On when TRUST_PROXY_HEADERS=1, and automatically on a
    HuggingFace Space (HF sets SPACE_ID in the container; the Space is only reachable through
    HF's proxy). Off otherwise: a directly-exposed server must not believe a client-supplied
    header. Caveat: the first X-Forwarded-For entry is client-controllable, so a hostile
    caller can rotate it to dodge the limit -- it protects against loops and accidents, not a
    determined attacker."""
    return os.environ.get("TRUST_PROXY_HEADERS", "0") == "1" or bool(os.environ.get("SPACE_ID"))


def intent_router_artifact_override() -> str | None:
    """Evaluate / run a candidate router WITHOUT promoting it. Promotion to production
    needs a human sign-off (mlops/intent_router/registry.py); this does not bypass it,
    it only points one process at a different file."""
    return os.environ.get("INTENT_ROUTER_ARTIFACT") or None


def llm_price_per_1k_tokens() -> tuple[float, float] | None:
    """(input, output) USD per 1K tokens for the CURRENT model, from LLM_PRICE_INPUT_PER_1K /
    LLM_PRICE_OUTPUT_PER_1K. None when unset: cost is then reported as unknown rather than
    guessed from another provider's price list."""
    try:
        return (
            float(os.environ["LLM_PRICE_INPUT_PER_1K"]),
            float(os.environ["LLM_PRICE_OUTPUT_PER_1K"]),
        )
    except (KeyError, ValueError):
        return None
