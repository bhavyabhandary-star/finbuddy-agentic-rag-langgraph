"""Fast, stubbed-provider tests of generate_node's wrong-script retry/fallback
branch. Deliberately separate from test_generate_integration.py, which is
real-provider-only by its own docstring -- this file tests OUR branching
logic (retry once, then use the fixed fallback), not a provider's actual
generation quality, which is already verified against real production calls
(see guardrails/output_guardrails.py's is_expected_script docstring).
"""
from __future__ import annotations

from agent.nodes.generate import WRONG_SCRIPT_FALLBACK_ANSWER, generate_node

GOOD_HINDI_JSON = '{"answer": "आपका क्रेडिट स्कोर अच्छा है।", "sources": [], "confidence": 0.9, "escalate_to_human": false}'
WRONG_SCRIPT_JSON = '{"answer": "Your credit score is good.", "sources": [], "confidence": 0.9, "escalate_to_human": false}'

POLICY_STATE = {
    "query": "why do you need my UPI data",
    "route": "policy",
    "retrieved_chunks": [],
    "top_score": 0.9,
    "sufficient": True,
    "response_language": "hi",
}


class _StubProvider:
    """Returns each of `responses` in order, one per call to complete()."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        response = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return response


def test_generate_node_accepts_correct_script_without_retry():
    provider = _StubProvider([GOOD_HINDI_JSON])
    result = generate_node(POLICY_STATE, provider=provider)
    assert result["answer"] == "आपका क्रेडिट स्कोर अच्छा है।"
    assert provider.call_count == 1


def test_generate_node_retries_once_and_recovers_on_wrong_script():
    provider = _StubProvider([WRONG_SCRIPT_JSON, GOOD_HINDI_JSON])
    result = generate_node(POLICY_STATE, provider=provider)
    assert result["answer"] == "आपका क्रेडिट स्कोर अच्छा है।"
    assert provider.call_count == 2


def test_generate_node_falls_back_to_fixed_answer_after_failed_retry():
    provider = _StubProvider([WRONG_SCRIPT_JSON, WRONG_SCRIPT_JSON])
    result = generate_node(POLICY_STATE, provider=provider)
    assert result["answer"] == WRONG_SCRIPT_FALLBACK_ANSWER["hi"]
    assert provider.call_count == 2


def test_generate_node_never_triggers_script_check_for_english():
    state = {**POLICY_STATE, "response_language": "en"}
    provider = _StubProvider([WRONG_SCRIPT_JSON])
    result = generate_node(state, provider=provider)
    assert result["answer"] == "Your credit score is good."
    assert provider.call_count == 1


def test_generate_node_uses_fixed_fallback_for_kannada_without_calling_provider():
    # Kannada is unconditional -- even a provider that would happily return a
    # perfectly-scripted Kannada response should never be asked, since real
    # testing showed correct-script Kannada from this project's provider is
    # still incoherent gibberish (see generate_node's kn branch docstring).
    state = {**POLICY_STATE, "response_language": "kn"}
    provider = _StubProvider(['{"answer": "ನಿಮ್ಮ ಕ್ರೆಡಿಟ್ ಸ್ಕೋರ್ ಉತ್ತಮವಾಗಿದೆ.", "sources": [], "confidence": 0.9, "escalate_to_human": false}'])
    result = generate_node(state, provider=provider)
    assert result["answer"] == WRONG_SCRIPT_FALLBACK_ANSWER["kn"]
    assert provider.call_count == 0


def test_generate_node_kannada_fallback_still_honors_classical_escalation():
    # The kn shortcut bypasses the model, but must not bypass the classical
    # sufficiency override -- an insufficient-context query should still
    # escalate, same as English/Hindi.
    state = {**POLICY_STATE, "response_language": "kn", "sufficient": False}
    provider = _StubProvider(["unused"])
    result = generate_node(state, provider=provider)
    assert result["escalate_to_human"] is True
    assert provider.call_count == 0
