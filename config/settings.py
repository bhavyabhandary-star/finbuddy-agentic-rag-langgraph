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
MAX_OUTPUT_TOKENS = 512  # explicit cap per kickoff_prompt.md's cost/latency section

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

# ---- Evaluation -------------------------------------------------------------
# Below this fraction of scenarios routed correctly, `eval.evaluate_agent --gate`
# fails the build. (This is the only CI quality gate today.)
GATE_MIN_ROUTING_ACCURACY = 0.75


# ---- Call-time env getters (secrets: no defaults) ---------------------------
def llm_provider_name() -> str:
    return os.environ.get("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()


def hf_model() -> str:
    return os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)


def hf_provider() -> str:
    return os.environ.get("HF_PROVIDER", DEFAULT_HF_PROVIDER)


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
