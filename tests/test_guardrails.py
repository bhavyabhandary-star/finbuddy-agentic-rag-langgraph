from guardrails.input_guardrails import contains_pii, sanitize_retrieved_text
from guardrails.output_guardrails import is_expected_script, validate_structured_output
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


def test_is_expected_script_passes_english_regardless_of_content():
    assert is_expected_script("Yes, you can get a loan.", "en") is True


def test_is_expected_script_passes_correct_devanagari_hindi():
    assert is_expected_script("आपका क्रेडिट स्कोर अच्छा है।", "hi") is True


def test_is_expected_script_passes_correct_kannada_script():
    assert is_expected_script("ನಿಮ್ಮ ಕ್ರೆಡಿಟ್ ಸ್ಕೋರ್ ಉತ್ತಮವಾಗಿದೆ.", "kn") is True


def test_is_expected_script_fails_english_answer_for_kannada_request():
    # Real production example: a kn request answered almost entirely in
    # English with stray Han/Bopomofo characters prepended.
    text = "ㄧ过关isable consent for using personal data should be free, specific, informed, and clear."
    assert is_expected_script(text, "kn") is False


def test_is_expected_script_fails_devanagari_answer_for_kannada_request():
    # Real production example: a kn request answered in Hindi (Devanagari),
    # not Kannada script at all.
    text = "डिफॉल्ट लॉस गारंटी कवर की सीमा राशी दो करोड़ रुपये हो सकती है।"
    assert is_expected_script(text, "kn") is False


def test_is_expected_script_tolerates_a_few_english_loanwords_in_hindi():
    assert is_expected_script("आपका UPI खाता सक्रिय है और FinBuddy इसे देख सकता है।", "hi") is True
