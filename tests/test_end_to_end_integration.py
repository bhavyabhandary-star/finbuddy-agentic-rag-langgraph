"""Real, non-mocked end-to-end tests of the FULL compiled LangGraph state
machine — every layer for real: Intent Router (trained model) -> real Chroma
retrieval + cross-encoder re-ranking -> real credit-assessment tools (live
production scoring API + real Risk-Trend artifact) -> real LLM generation.

Milestone step 6: verifies the wiring between all previously-verified pieces,
not any single piece in isolation (those have their own dedicated tests).

Skipped automatically when its real dependency isn't available: HF_TOKEN for
generation, the ingested Chroma corpus for the policy path, the Risk-Trend
artifact for the credit-assessment path. Requires network access (the live
scoring API, HF Inference Providers).

Run: pytest tests/test_end_to_end_integration.py -v -s
"""
from __future__ import annotations

import os

import pytest

from agent.graph import build_graph
from agent.nodes.generate import HuggingFaceProvider
from ingestion.setu_feed import CACHED_PROFILE_PATH
from tools.credit_tools import RISK_TREND_ARTIFACT_PATH

requires_hf_token = pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"), reason="requires HF_TOKEN in .env for generation"
)
requires_chroma_corpus = pytest.mark.skipif(
    not os.path.exists("./chroma_data"), reason="requires the real PDF corpus ingested (python -m ingestion.build_corpus)"
)
requires_risk_trend_artifact = pytest.mark.skipif(
    not os.path.exists(RISK_TREND_ARTIFACT_PATH), reason="requires risk_trend_logreg.joblib — see README setup"
)
requires_setu_profile = pytest.mark.skipif(
    not os.path.exists(CACHED_PROFILE_PATH), reason="requires data/setu_real_profiles.jsonl — see README setup"
)

# Real z-scored delta features, sampled from finbuddy-project's actual
# training data (see tests/test_tools.py) — matches assess_risk_trend's
# verified contract (pre-z-scored, not raw natural-unit deltas).
_REAL_IMPROVING_DELTA = {
    "delta_avg_monthly_income": 0.553392,
    "delta_income_regularity_score": 1.68956,
    "delta_tx_count_30d": 0.94078,
    "delta_merchant_diversity": 0.926998,
    "delta_balance_dip_frequency": -1.809724,
    "delta_b2b_ratio": 0.191642,
    "delta_avg_transaction_size": 0.0,
    "delta_tenure_months": 0.0,
}


def _safe_print(*args) -> None:
    """Windows consoles default to cp1252, which can't render every character
    a real LLM response produces (smart quotes, em-dashes) — this crashed a
    test's print() call once, not the pipeline itself. Encode-safe for
    visibility here; the actual API/UI layers use UTF-8 and don't need this.
    """
    text = " ".join(str(a) for a in args)
    print(text.encode("ascii", "replace").decode("ascii"))


@requires_hf_token
def test_off_topic_query_full_graph_escalates_without_llm_call():
    """No corpus/artifact dependency — the off-topic path never touches them."""
    graph = build_graph(provider=HuggingFaceProvider())
    result = graph.invoke({"query": "what's the weather like today", "session_id": "e2e-off-topic"})

    _safe_print("\n--- off-topic (full graph) ---")
    _safe_print("answer:", result["answer"])
    _safe_print("escalate_to_human:", result["escalate_to_human"])

    assert result["escalate_to_human"] is True
    assert result["route"] == "off_topic"


@requires_hf_token
@requires_chroma_corpus
def test_policy_query_full_graph_real_retrieval_and_generation():
    graph = build_graph(provider=HuggingFaceProvider())
    result = graph.invoke(
        {"query": "who is eligible to register as an account aggregator NBFC", "session_id": "e2e-policy"}
    )

    _safe_print("\n--- policy (full graph) ---")
    _safe_print("route:", result["route"], "| route_confidence:", result.get("route_confidence"))
    _safe_print("top_score:", result.get("top_score"), "| sufficient:", result.get("sufficient"))
    _safe_print("answer:", result["answer"])
    _safe_print("sources:", result["sources"])
    _safe_print("escalate_to_human:", result["escalate_to_human"])

    assert result["route"] == "policy"
    assert result["sufficient"] is True
    assert result["escalate_to_human"] is False
    assert len(result["answer"]) > 0
    assert any("RBI_AA_Master_Direction" in s for s in result["sources"])


@requires_hf_token
@requires_risk_trend_artifact
def test_credit_assessment_query_full_graph_real_api_and_model():
    graph = build_graph(provider=HuggingFaceProvider())
    result = graph.invoke(
        {
            "query": "can you check my credit score and tell me my financial trend",
            "session_id": "e2e-credit",
            "credit_signals": {
                "avg_monthly_income": 21000,
                "income_regularity_score": 0.81,
                "tx_count_30d": 340,
                "merchant_diversity": 11,
                "balance_dip_frequency": 3,
                "b2b_ratio": 0.12,
                "avg_transaction_size": 95,
                "tenure_months": 6,
            },
            "risk_trend_delta_features": _REAL_IMPROVING_DELTA,
        }
    )

    _safe_print("\n--- credit_assessment (full graph) ---")
    _safe_print("route:", result["route"])
    _safe_print("credit_assessment:", result.get("credit_assessment"))
    _safe_print("risk_trend:", result.get("risk_trend"))
    _safe_print("answer:", result["answer"])
    _safe_print("disclaimer:", result.get("disclaimer"))
    _safe_print("escalate_to_human:", result["escalate_to_human"])

    assert result["route"] == "credit_assessment"
    credit = result["credit_assessment"]
    assert credit.tool_error is None  # the live API call must have actually succeeded
    assert credit.credit_score > 0
    assert result["risk_trend"].trend == "improving"
    assert result["disclaimer"] is not None  # fixed, code-appended — never LLM-generated
    assert result["escalate_to_human"] is False


@requires_hf_token
@requires_setu_profile
def test_setu_feed_real_signals_through_api_layer():
    """Real, non-mocked test of the capstone's own "Setu AA Feed -> FastAPI
    Inference" pipeline stage: the API layer's use_setu_feed=True flag pulls
    the real, already-consented Setu sandbox profile and feeds ITS signals
    into the live scoring API — not hand-typed values. This is a different
    code path from the other credit-assessment test above (which passes
    credit_signals directly), so it needs its own real run to verify.
    """
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        response = client.post(
            "/agent/run",
            json={
                "query": "can you check my credit score",
                "session_id": "e2e-setu-feed",
                "use_setu_feed": True,
            },
        )

    _safe_print("\n--- Setu AA feed -> API layer (real) ---")
    _safe_print("status:", response.status_code)
    _safe_print("body:", response.json())

    assert response.status_code == 200
    body = response.json()
    assert body["escalate_to_human"] is False
    assert len(body["answer"]) > 0
    assert body["disclaimer"] is not None
