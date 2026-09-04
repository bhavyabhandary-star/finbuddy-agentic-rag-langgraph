from guardrails.input_guardrails import contains_pii, sanitize_retrieved_text
from guardrails.output_guardrails import validate_structured_output
from guardrails.tool_policy import SideEffectNotConfirmedError, require_confirmation, safe_tool_call
from tools.schemas import AgentResponse


def test_sanitize_strips_control_characters():
    dirty = "hello\x00world"
    assert sanitize_retrieved_text(dirty) == "helloworld"


def test_contains_pii_detects_mobile_number():
    assert contains_pii("call me at 9876543210") is True


def test_contains_pii_false_on_clean_text():
    assert contains_pii("what is the RBI data retention rule") is False


def test_safe_tool_call_returns_fallback_on_failure():
    def _always_fails():
        raise RuntimeError("boom")

    result = safe_tool_call(_always_fails, tool_name="test_tool", fallback=lambda err: f"fallback:{err}")
    assert result == "fallback:boom"


def test_safe_tool_call_returns_value_on_success():
    result = safe_tool_call(lambda: "ok", tool_name="test_tool", fallback=lambda err: "unused")
    assert result == "ok"


def test_require_confirmation_raises_without_confirmation():
    try:
        require_confirmation("send_message", user_confirmed=False)
        assert False, "expected SideEffectNotConfirmedError"
    except SideEffectNotConfirmedError:
        pass


def test_validate_structured_output_succeeds_on_valid_json():
    raw = '{"answer": "hi", "sources": [], "confidence": 0.9, "escalate_to_human": false}'
    result = validate_structured_output(raw, AgentResponse, regenerate=lambda err: raw)
    assert result.answer == "hi"


def test_validate_structured_output_retries_and_recovers():
    bad = '{"answer": "hi"}'  # missing required "confidence"
    good = '{"answer": "hi", "sources": [], "confidence": 0.5, "escalate_to_human": false}'
    result = validate_structured_output(bad, AgentResponse, regenerate=lambda err: good)
    assert result.confidence == 0.5
