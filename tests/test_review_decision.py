"""The credit decision is the LLM's, made from the whole profile and validated against the scoring
evidence. The validator never writes a decision: it refuses one that contradicts the evidence, retries
once with the reason, and otherwise fails closed to a human. (Real-model behaviour is checked in
tests/test_decision_eval_real.py.)"""
from __future__ import annotations

import json

import pytest

from agent.nodes.generate import (
    DECISION_UNVERIFIED_MESSAGE,
    PROFILE_SIGNAL_KEYS,
    _build_data_section,
    generate_node,
)
from observability.tracing import RunTracer
from prompts import ACTIVE_VERSIONS, load_prompt
from tools.schemas import CreditAssessmentResult

SIGNALS = {
    "avg_monthly_income": 37000, "income_regularity_score": 0.91, "tx_count_30d": 900, "merchant_diversity": 28,
    "balance_dip_frequency": 0, "b2b_ratio": 0.65, "avg_transaction_size": 210, "tenure_months": 12,
    "geography": "urban", "gender": "female", "pincode": "600001",  # must never reach the model
}


def _credit(approved=True, is_anomalous=False, tool_error=None, score=850) -> CreditAssessmentResult:
    return CreditAssessmentResult(
        credit_score=score, calibrated_probability_of_repayment=0.99, approved=approved,
        fairness_mitigation_applied=True, income_band="medium", is_anomalous=is_anomalous,
        anomaly_note="Unusual UPI pattern detected pre-scoring" if is_anomalous else None,
        top_3_factors=[], latency_ms=10.0, tool_error=tool_error,
    )


def _state(credit, language="en"):
    return {"query": "can you check my credit score", "route": "credit_assessment", "credit_assessment": credit,
            "credit_signals": SIGNALS, "risk_trend": None, "response_language": language}


def _reply(decision, escalate=False, answer="ok"):
    return json.dumps({"answer": answer, "decision": decision, "confidence": 0.8, "escalate_to_human": escalate})


class _Scripted:
    """Returns the scripted replies in order and records every prompt it was sent."""
    last_usage = None

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system_prompt, user_query, max_tokens):
        self.prompts.append(system_prompt)
        return self.replies.pop(0)


def test_model_review_on_a_flagged_profile_is_used_as_is_and_handed_to_a_person():
    p = _Scripted(_reply("review", escalate=True, answer="This needs a person to review: the pattern is unusual."))
    out = generate_node(_state(_credit(is_anomalous=True)), provider=p)
    assert out["decision"] == "review"
    assert out["answer"] == "This needs a person to review: the pattern is unusual."  # the model's words, not fixed text
    assert out["escalate_to_human"] is True and out["failure_reason"] is None
    assert len(p.prompts) == 1 and out["disclaimer"]


def test_review_is_always_a_handoff_even_if_the_model_forgot_to_set_the_flag():
    out = generate_node(_state(_credit(is_anomalous=True)), provider=_Scripted(_reply("review", escalate=False)))
    assert out["decision"] == "review" and out["escalate_to_human"] is True


def test_model_approval_on_a_flagged_profile_gets_one_corrective_retry_then_is_accepted_if_fixed():
    p = _Scripted(_reply("approved", answer="You can get a loan."), _reply("review", answer="A person should review this."))
    out = generate_node(_state(_credit(is_anomalous=True)), provider=p)
    assert out["decision"] == "review" and out["answer"] == "A person should review this."
    assert "decision_inconsistent" in out["guardrail_flags"] and "decision_unverified" not in out["guardrail_flags"]
    assert "flagged as unusual" in p.prompts[1]  # the retry told the model exactly what was wrong


def test_model_that_keeps_approving_a_flagged_profile_fails_closed_to_a_person():
    p = _Scripted(_reply("approved"), _reply("approved"))
    out = generate_node(_state(_credit(is_anomalous=True)), provider=p)
    assert out["decision"] == "review" and out["escalate_to_human"] is True
    assert out["answer"] == DECISION_UNVERIFIED_MESSAGE["en"]
    assert out["failure_reason"] == "decision_unverified" and "decision_unverified" in out["guardrail_flags"]


def test_model_approval_when_the_scoring_service_did_not_approve_is_rejected():
    p = _Scripted(_reply("approved"), _reply("not_approved", answer="Not approved yet."))
    out = generate_node(_state(_credit(approved=False, score=520)), provider=p)
    assert out["decision"] == "not_approved" and "decision_inconsistent" in out["guardrail_flags"]


def test_missing_decision_on_a_credit_assessment_is_rejected_and_fails_closed():
    p = _Scripted(_reply(None), _reply(None))
    out = generate_node(_state(_credit()), provider=p)
    assert out["decision"] == "review" and out["failure_reason"] == "decision_unverified"


def test_clean_profile_approval_passes_untouched():
    out = generate_node(_state(_credit(score=770)), provider=_Scripted(_reply("approved", answer="Approved.")))
    assert out["decision"] == "approved" and out["escalate_to_human"] is False and out["guardrail_flags"] == []


def test_decision_is_none_off_the_credit_route():
    state = {"query": "q", "route": "policy", "retrieved_chunks": [], "sufficient": True, "response_language": "en"}
    out = generate_node(state, provider=_Scripted(_reply(None)))
    assert out["decision"] is None


def test_hindi_fail_closed_message_is_hindi():
    out = generate_node(_state(_credit(is_anomalous=True), "hi"), provider=_Scripted(_reply("approved"), _reply("approved")))
    assert out["answer"] == DECISION_UNVERIFIED_MESSAGE["hi"] and out["decision"] == "review"


def test_kannada_fixed_fallback_on_a_credit_assessment_is_a_review_not_an_approval():
    class _NoCall:
        last_usage = None
        def complete(self, *a, **k):  # pragma: no cover
            raise AssertionError("Kannada never calls the model")
    out = generate_node(_state(_credit(), "kn"), provider=_NoCall())
    assert out["decision"] == "review" and out["escalate_to_human"] is True


def test_model_sees_the_whole_profile_but_only_the_eight_signals():
    data = json.loads(_build_data_section(_state(_credit(is_anomalous=True))))
    assert data["credit_assessment"]["is_anomalous"] is True and data["credit_assessment"]["credit_score"] == 850
    assert set(data["profile_signals"]) == set(PROFILE_SIGNAL_KEYS)
    for banned in ("gender", "pincode", "geography"):
        assert banned not in data["profile_signals"]


def test_decision_and_check_are_in_the_audit_trace_with_the_prompt_version():
    tracer = RunTracer(run_id="t")
    generate_node(_state(_credit(is_anomalous=True)), provider=_Scripted(_reply("review")), tracer=tracer)
    check = [s for s in tracer.steps if s["node"] == "decision_check"][0]
    assert check["llm_decision"] == "review" and check["prompt_version"] == ACTIVE_VERSIONS["generate_system"] == "v2"
    gen = [s for s in tracer.steps if s["node"] == "generate"][0]
    assert gen["llm_decision"] == "review"


def test_the_prompt_tells_the_model_to_weigh_the_anomaly_flag_and_not_to_approve_flagged_profiles():
    text = " ".join(load_prompt("generate_system").split())  # collapse the prompt's line wraps
    assert "is_anomalous" in text and 'Never choose "approved" for a flagged profile' in text
    assert "decision support" in text


def test_a_review_explanation_that_praises_approval_is_refused_and_retried():
    p = _Scripted(
        _reply("review", answer="Your score supports approval, but a person will check it."),
        _reply("review", answer="Your application is under review. A person from our team will take a closer look."),
    )
    out = generate_node(_state(_credit(is_anomalous=True)), provider=p)
    assert out["answer"].startswith("Your application is under review")
    assert "decision_inconsistent" in out["guardrail_flags"] and "decision_unverified" not in out["guardrail_flags"]
    assert "talks about approval" in p.prompts[1]


def test_a_review_explanation_that_keeps_praising_approval_fails_closed():
    bad = _reply("review", answer="The scoring service approved your application, but a person will check it.")
    out = generate_node(_state(_credit(is_anomalous=True)), provider=_Scripted(bad, bad))
    assert out["answer"] == DECISION_UNVERIFIED_MESSAGE["en"] and out["decision"] == "review"


@pytest.mark.parametrize(
    "answer",
    [
        "Your application is under review and a person from our team will check it before anything is decided.",
        "You are not approved yet. Your income varies a lot from month to month.",
        "We have not approved your application yet.",
        "The scoring service did not approve this.",
    ],
)
def test_ordinary_review_and_decline_wording_is_not_mistaken_for_an_approval_claim(answer):
    from agent.nodes.generate import _APPROVAL_CLAIM
    assert not _APPROVAL_CLAIM.search(answer)


@pytest.mark.parametrize(
    "answer",
    ["The score supports approval.", "You can get a loan.", "You should be able to get a loan.",
     "You're eligible for a limit.", "The scoring service approved your application.", "Your loan is approved."],
)
def test_approval_claims_are_recognised(answer):
    from agent.nodes.generate import _APPROVAL_CLAIM
    assert _APPROVAL_CLAIM.search(answer)
