"""Real, non-mocked tests of the Generate node against actual LLM providers.

Each provider's tests are skipped independently when its credential isn't
present (ANTHROPIC_API_KEY for Claude, HF_TOKEN for the Hugging Face Inference
Providers fallback) — so this file runs against whichever providers you've
actually configured, and never breaks CI when neither is set.

Run: pytest tests/test_generate_integration.py -v -s
"""
from __future__ import annotations

import os

import pytest

from agent.nodes.generate import ClaudeProvider, HuggingFaceProvider, generate_node
from tools.schemas import CreditAssessmentResult, RetrievedChunk, ScoreFactor

PROVIDERS = [
    pytest.param(
        "claude",
        ClaudeProvider,
        marks=pytest.mark.skipif(
            not os.environ.get("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY in .env"
        ),
    ),
    pytest.param(
        "huggingface",
        HuggingFaceProvider,
        marks=pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="requires HF_TOKEN in .env"),
    ),
]


@pytest.mark.parametrize("provider_name, provider_cls", PROVIDERS)
def test_generate_node_policy_path_real_call(provider_name, provider_cls):
    state = {
        "query": "why do you need my UPI transaction data",
        "route": "policy",
        "retrieved_chunks": [
            RetrievedChunk(
                text=(
                    "FinBuddy needs your UPI transaction data to build an income story, "
                    "as most gig workers don't have a traditional credit history."
                ),
                source="why_upi_data_needed.md",
                score=0.91,
            )
        ],
        "top_score": 0.91,
        "sufficient": True,
    }

    result = generate_node(state, provider=provider_cls())

    print(f"\n--- Generate node output (policy path, provider={provider_name}) ---")
    print("answer:", result["answer"])
    print("sources:", result["sources"])
    print("confidence:", result["confidence"])
    print("escalate_to_human:", result["escalate_to_human"])

    assert isinstance(result["answer"], str) and len(result["answer"]) > 0
    assert isinstance(result["confidence"], float)
    assert isinstance(result["escalate_to_human"], bool)


@pytest.mark.parametrize("provider_name, provider_cls", PROVIDERS)
def test_generate_node_credit_assessment_path_real_call(provider_name, provider_cls):
    state = {
        "query": "can you check my credit score",
        "route": "credit_assessment",
        "credit_assessment": CreditAssessmentResult(
            credit_score=738,
            calibrated_probability_of_repayment=0.81,
            approved=True,
            fairness_mitigation_applied=True,
            income_band="middle",
            is_anomalous=False,
            anomaly_note=None,
            top_3_factors=[
                ScoreFactor(
                    factor="income_regularity_score",
                    shap_contribution=0.12,
                    direction="positive",
                    plain_english="Your income arrives on a regular schedule.",
                    action="Keep receiving payments through the same UPI handle.",
                )
            ],
            latency_ms=85.0,
        ),
        "risk_trend": None,
    }

    result = generate_node(state, provider=provider_cls())

    print(f"\n--- Generate node output (credit_assessment path, provider={provider_name}) ---")
    print("answer:", result["answer"])
    print("confidence:", result["confidence"])

    assert isinstance(result["answer"], str) and len(result["answer"]) > 0
    # A real credit_score of 738 exists in DATA — the answer shouldn't need to escalate.
    assert result["escalate_to_human"] is False


@pytest.mark.parametrize("provider_name, provider_cls", PROVIDERS)
def test_generate_node_escalates_on_insufficient_context(provider_name, provider_cls):
    """An off-corpus-shaped query with no useful retrieved context should set
    escalate_to_human=True rather than let the model guess — same grounding
    gate production FinBuddy already proves works.
    """
    state = {
        "query": "what stock should I invest in this week",
        "route": "policy",
        "retrieved_chunks": [],
        "top_score": 0.0,
        "sufficient": False,
    }

    result = generate_node(state, provider=provider_cls())

    print(f"\n--- Generate node output (insufficient context, provider={provider_name}) ---")
    print("answer:", result["answer"])
    print("escalate_to_human:", result["escalate_to_human"])

    assert result["escalate_to_human"] is True
