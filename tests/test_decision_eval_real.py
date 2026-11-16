"""Evaluation of the REAL model's credit decisions on golden profiles (the model is prompted, not
fine-tuned, so this suite is how its decision behaviour is measured and regression-tested).

Runs against whichever provider is configured (ANTHROPIC_API_KEY or HF_TOKEN) and is skipped
otherwise, so it never breaks CI without keys. Each case asserts the decision AND that the explanation
agrees with it (no approval language under a review/not_approved decision).

Run: pytest tests/test_decision_eval_real.py -v -s
"""
from __future__ import annotations

import os

import pytest

from agent.nodes.generate import ClaudeProvider, HuggingFaceProvider, generate_node
from tools.schemas import CreditAssessmentResult, ScoreFactor

PROVIDERS = [
    pytest.param("claude", ClaudeProvider, marks=pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")),
    pytest.param("huggingface", HuggingFaceProvider, marks=pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="needs HF_TOKEN")),
]

FACTOR = ScoreFactor(factor="income_regularity_score", shap_contribution=1.0, direction="positive",
                     plain_english="Your income arrives on a regular monthly pattern.", action="Keep it regular.")

RAJESH = {"avg_monthly_income": 20000, "income_regularity_score": 0.82, "tx_count_30d": 400, "merchant_diversity": 12,
          "balance_dip_frequency": 2, "b2b_ratio": 0.15, "avg_transaction_size": 50, "tenure_months": 6}
MEENA = {"avg_monthly_income": 37000, "income_regularity_score": 0.91, "tx_count_30d": 900, "merchant_diversity": 28,
         "balance_dip_frequency": 0, "b2b_ratio": 0.65, "avg_transaction_size": 210, "tenure_months": 12}
WEAK = {"avg_monthly_income": 6000, "income_regularity_score": 0.2, "tx_count_30d": 25, "merchant_diversity": 2,
        "balance_dip_frequency": 9, "b2b_ratio": 0.0, "avg_transaction_size": 120, "tenure_months": 2}


def _credit(score, prob, approved, anomalous):
    return CreditAssessmentResult(
        credit_score=score, calibrated_probability_of_repayment=prob, approved=approved, fairness_mitigation_applied=True,
        income_band="medium", is_anomalous=anomalous,
        anomaly_note=("Unusual UPI pattern detected pre-scoring -- score should be treated as lower-confidence and may "
                      "warrant manual review.") if anomalous else None,
        top_3_factors=[FACTOR], latency_ms=10.0)


GOLDEN = [
    pytest.param("rajesh_clean_approved", _credit(770, 0.86, True, False), RAJESH, "approved", id="rajesh_clean_approved"),
    pytest.param("meena_flagged_high_score", _credit(850, 0.99, True, True), MEENA, "review", id="meena_flagged_high_score"),
    pytest.param("weak_declined", _credit(480, 0.18, False, False), WEAK, "not_approved", id="weak_declined"),
]

from agent.nodes.generate import _APPROVAL_CLAIM as APPROVAL_LANGUAGE  # one source of truth with the validator


@pytest.mark.parametrize("provider_name, provider_cls", PROVIDERS)
@pytest.mark.parametrize("name, credit, signals, expected", GOLDEN)
def test_real_model_decides_correctly_and_explains_consistently(provider_name, provider_cls, name, credit, signals, expected):
    state = {"query": "can you check my credit score", "route": "credit_assessment", "credit_assessment": credit,
             "credit_signals": signals, "risk_trend": None, "response_language": "en"}
    out = generate_node(state, provider=provider_cls())
    if out.get("failure_reason") == "llm_unavailable":
        pytest.skip(f"{provider_name} unavailable")
    print(f"\n[{provider_name}] {name}: decision={out['decision']} escalate={out['escalate_to_human']} flags={out['guardrail_flags']}\n  {out['answer']}")
    assert out["decision"] == expected, f"model decided {out['decision']!r}, expected {expected!r}: {out['answer']}"
    assert out["escalate_to_human"] is (expected == "review")
    assert out["failure_reason"] is None, "the model needed the fail-closed path"
    if expected != "approved":
        assert not APPROVAL_LANGUAGE.search(out["answer"]), f"explanation reads as an approval: {out['answer']}"
