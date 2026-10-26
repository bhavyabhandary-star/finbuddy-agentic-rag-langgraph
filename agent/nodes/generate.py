"""Generate node — the ONLY mandatory LLM call in the whole graph.

Structured Role/Instructions/Constraints/Data/Output-format prompt (Session 14,
slide 31), forced Pydantic-validated JSON, max_tokens set, provider-agnostic
(Claude default, Ollama fallback verified once — see kickoff_prompt.md).
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod

from pydantic import ValidationError

from agent.state import AgentState
from config import settings
from config.settings import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_HF_MODEL,
    LANGUAGE_NAMES,
    LLM_TIMEOUT_SECONDS,
    MAX_OUTPUT_TOKENS,
)
from guardrails.grounding import check_grounding
from guardrails.input_guardrails import sanitize_retrieved_text, sanitize_with_flags
from guardrails.output_guardrails import is_expected_script, validate_structured_output
from observability.tracing import RunTracer
from prompts import ACTIVE_VERSIONS, load_prompt, render_retry_suffix, render_system_prompt
from tools.schemas import AgentResponse

# Tunables live in config/settings.py; prompts live in prompts/ (versioned).
# Re-exported here under the same names so existing imports keep working.
SYSTEM_PROMPT_TEMPLATE = load_prompt("generate_system")


class LLMUnavailableError(Exception):
    """The LLM provider failed (outage, timeout, rate limit, auth, network). generate_node
    turns this into a clean human escalation instead of letting it become an HTTP 500."""


class LLMProvider(ABC):
    # (input_tokens, output_tokens) of the most recent call when the provider reports it;
    # None means "not reported" and the caller falls back to a labelled estimate.
    last_usage: tuple[int, int] | None = None

    @abstractmethod
    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        """Returns raw completion text (expected to be JSON per the schema)."""


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL):
        self.model = model

    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=self.model, max_tokens=max_tokens, timeout=LLM_TIMEOUT_SECONDS)
        response = llm.invoke([("system", system_prompt), ("human", user_query)])
        usage = getattr(response, "usage_metadata", None) or {}
        self.last_usage = (usage["input_tokens"], usage["output_tokens"]) if usage else None
        return response.content


class OllamaProvider(LLMProvider):
    """Local-model fallback — verify once with a real run, keep the proof, but
    this is not the default demo path (see build_prompt.md's resolved decisions).
    """

    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        import ollama

        model = settings.ollama_model()
        response = ollama.Client(timeout=LLM_TIMEOUT_SECONDS).chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            options={"num_predict": max_tokens},
        )
        counts = (response.get("prompt_eval_count") or 0, response.get("eval_count") or 0)
        self.last_usage = counts if any(counts) else None
        return response["message"]["content"]


class HuggingFaceProvider(LLMProvider):
    """Hosted local-model-family fallback via HF's Inference Providers (not
    Claude, not a locally-run process) — an alternative to OllamaProvider for
    when a paid HF subscription is available instead of local Ollama install.

    provider="auto" failed on this account with "not supported by any
    provider you have enabled" (verified by running it) — the account has no
    default/enabled provider for auto-routing, so an explicit provider is
    required. "featherless-ai" was verified working for DEFAULT_HF_MODEL;
    override via HF_PROVIDER if you enable a different one in your HF
    settings (huggingface.co/settings/inference-providers).
    """

    def __init__(self, model: str | None = None, provider: str | None = None):
        self.model = model or settings.hf_model()
        self.provider = provider or settings.hf_provider()

    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            model=self.model, provider=self.provider, token=settings.hf_token(), timeout=LLM_TIMEOUT_SECONDS
        )
        # Verified necessary, not optional: without response_format, this model
        # (unlike Claude) frequently prepends prose before the JSON despite an
        # explicit "return ONLY JSON" instruction, failing schema validation.
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        usage = getattr(response, "usage", None)
        self.last_usage = (usage.prompt_tokens, usage.completion_tokens) if usage else None
        return response.choices[0].message.content


CREDIT_ASSESSMENT_DISCLAIMER = {
    "en": (
        "This is an AI-generated explanation of an automated assessment, not "
        "financial advice. Treat it as decision-support — confirm any important "
        "decision with FinBuddy support."
    ),
    "hi": (
        "यह एक स्वचालित मूल्यांकन का AI-जनित स्पष्टीकरण है, वित्तीय सलाह नहीं है। "
        "इसे केवल सहायक जानकारी मानें — कोई भी महत्वपूर्ण निर्णय लेने से पहले FinBuddy सहायता टीम से पुष्टि करें।"
    ),
    "kn": (
        "ಇದು ಸ್ವಯಂಚಾಲಿತ ಮೌಲ್ಯಮಾಪನದ AI-ಸೃಜಿತ ವಿವರಣೆಯಾಗಿದೆ, ಹಣಕಾಸಿನ ಸಲಹೆಯಲ್ಲ. "
        "ಇದನ್ನು ನಿರ್ಧಾರ-ಸಹಾಯವಾಗಿ ಮಾತ್ರ ಪರಿಗಣಿಸಿ — ಯಾವುದೇ ಮುಖ್ಯ ನಿರ್ಧಾರವನ್ನು FinBuddy ಬೆಂಬಲ ತಂಡದೊಂದಿಗೆ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ."
    ),
}

# Fixed, human-authored fallback answer for when the model's own generation
# fails the is_expected_script guardrail twice in a row -- never LLM-generated,
# same reasoning as CREDIT_ASSESSMENT_DISCLAIMER: a user-facing string that
# must be guaranteed to actually be in the right script can't depend on the
# same model that just failed to produce that script.
WRONG_SCRIPT_FALLBACK_ANSWER = {
    "hi": (
        "क्षमा करें, मुझे यकीन नहीं है कि मैं अभी इस भाषा में सही जवाब दे पाऊंगी। "
        "कृपया अंग्रेज़ी में पूछें — मैं मदद करने की पूरी कोशिश करूंगी।"
    ),
    "kn": (
        "ಕ್ಷಮಿಸಿ, ನಾನು ಈ ಭಾಷೆಯಲ್ಲಿ ಸರಿಯಾಗಿ ಉತ್ತರಿಸಬಲ್ಲೆ ಎಂದು ಖಚಿತವಾಗಿಲ್ಲ. "
        "ದಯವಿಟ್ಟು ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ಕೇಳಿ — ನಾನು ಸಹಾಯ ಮಾಡಲು ಪ್ರಯತ್ನಿಸುತ್ತೇನೆ."
    ),
}


# Fixed, human-authored strings (never LLM-generated, same reasoning as the disclaimers above).
# The hi/kn wording mirrors the existing tool-error escalation text in agent/graph.py; it has
# not been reviewed by a native speaker.
LLM_UNAVAILABLE_MESSAGE = {
    "en": (
        "I couldn't prepare a verified answer just now because the language model is not "
        "responding. Please try again shortly, or a human coach can help directly."
    ),
    "hi": (
        "भाषा मॉडल अभी जवाब नहीं दे रहा, इसलिए मैं अभी एक सत्यापित जवाब तैयार नहीं कर पा रही। "
        "कृपया थोड़ी देर बाद फिर कोशिश करें, या सीधे एक इंसान कोच से मदद लें।"
    ),
    "kn": (
        "ಭಾಷಾ ಮಾದರಿ ಈಗ ಪ್ರತಿಕ್ರಿಯಿಸುತ್ತಿಲ್ಲ, ಆದ್ದರಿಂದ ಪರಿಶೀಲಿತ ಉತ್ತರ ನೀಡಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲ. "
        "ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ, ಅಥವಾ ನೇರವಾಗಿ ಮಾನವ ಕೋಚ್ ಸಹಾಯ ಪಡೆಯಿರಿ."
    ),
}
# Only English policy answers are grounding-checked (see guardrails/grounding.py), so en only.
UNGROUNDED_ANSWER_MESSAGE = (
    "I couldn't verify that answer against the RBI and DPDP documents I have, so I won't state it "
    "as policy. Let me connect you with a human coach."
)


def _complete(provider: LLMProvider, system_prompt: str, user_query: str, tracer: RunTracer | None) -> str:
    """One LLM call, with provider failures normalised and usage / latency recorded."""
    provider.last_usage = None
    started = time.perf_counter()
    attempts = 1 + settings.MAX_LLM_CALL_RETRIES
    for attempt in range(1, attempts + 1):
        try:
            raw = provider.complete(system_prompt, user_query, MAX_OUTPUT_TOKENS)
            break
        except Exception as exc:  # noqa: BLE001 -- any provider failure (network, 429, 5xx, auth, timeout)
            if tracer:
                tracer.log_step(
                    "generate", decision="llm_call_failed", attempt=attempt, error=f"{type(exc).__name__}: {exc}"[:300]
                )
            if attempt == attempts:
                raise LLMUnavailableError(f"{type(exc).__name__}: {exc}") from exc
            time.sleep(settings.LLM_RETRY_BACKOFF_SECONDS)
    if tracer:
        tracer.cost.record_llm_call(
            prompt_text=system_prompt + user_query,
            completion_text=raw,
            reported_usage=provider.last_usage,
            seconds=time.perf_counter() - started,
        )
    return raw


def _deterministic_sources(state: AgentState) -> list[str]:
    """Real bug found end-to-end testing (milestone step 6): the prompt asked
    the LLM to fill in "sources" itself, and it populated the field with
    quoted excerpt text instead of document identifiers — a reasonable
    guess given the prompt never specified the format, but wrong, and not a
    failure mode worth prompt-engineering around when the actual retrieved
    chunks' source filenames are already known deterministically in state.
    Citation accuracy shouldn't depend on the LLM correctly copying it —
    same reasoning as the escalate_to_human override above.
    """
    if state.get("route") != "policy":
        return []
    chunks = state.get("retrieved_chunks", [])
    seen: list[str] = []
    for c in chunks:
        if c.source not in seen:
            seen.append(c.source)
    return seen


def _build_data_section(state: AgentState) -> str:
    if state.get("route") == "credit_assessment":
        credit = state.get("credit_assessment")
        trend = state.get("risk_trend")
        return json.dumps(
            {
                "credit_assessment": credit.model_dump() if credit else None,
                "risk_trend": trend.model_dump() if trend else None,
            }
        )
    chunks = state.get("retrieved_chunks", [])
    sanitized = [sanitize_retrieved_text(c.text) for c in chunks]
    return "\n---\n".join(sanitized) if sanitized else "(no relevant context retrieved)"


def _degraded(state: AgentState, reason: str, detail: str, tracer: RunTracer | None) -> AgentState:
    """Fail closed: a clean, fixed, human-escalating answer -- never a 500, never a guess."""
    language = state.get("response_language") or "en"
    if tracer:
        tracer.log_step("generate", decision=f"degraded: {reason}", error=detail[:300], routed_to_human=True)
    return {
        **state,
        "answer": LLM_UNAVAILABLE_MESSAGE.get(language, LLM_UNAVAILABLE_MESSAGE["en"]),
        "sources": [],
        "confidence": 0.0,
        "escalate_to_human": True,
        "disclaimer": None,
        "failure_reason": reason,
    }


def generate_node(
    state: AgentState, provider: LLMProvider | None = None, tracer: RunTracer | None = None
) -> AgentState:
    try:
        return _generate_core(state, provider, tracer)
    except LLMUnavailableError as exc:
        return _degraded(state, "llm_unavailable", str(exc), tracer)
    except ValidationError as exc:
        # The model kept returning output that is not valid AgentResponse JSON after the retry.
        return _degraded(state, "invalid_structured_output", str(exc), tracer)


def _generate_core(
    state: AgentState, provider: LLMProvider | None = None, tracer: RunTracer | None = None
) -> AgentState:
    provider = provider or ClaudeProvider()
    response_language = state.get("response_language") or "en"
    language_name = LANGUAGE_NAMES.get(response_language, "English")

    if response_language == "kn":
        # Unconditional fallback, not a retry target: real production calls
        # (including a fresh 3-run re-test after the hi/kn wrong-script retry
        # below was added) show this project's current provider doesn't just
        # occasionally pick the wrong script for Kannada -- even when it does
        # stay in Kannada Unicode script, the text is incoherent gibberish,
        # not real Kannada (stray Latin/Thai/Devanagari fragments mixed in).
        # is_expected_script can only catch wrong-script output, not
        # wrong-script-passing-but-meaningless content, so retrying and
        # hoping is pointless here -- skip generation entirely rather than
        # spend a real LLM call on output that's already known to be
        # unusable and would just get discarded.
        parsed: AgentResponse = AgentResponse(
            answer=WRONG_SCRIPT_FALLBACK_ANSWER["kn"], sources=[], confidence=0.0, escalate_to_human=False
        )
    else:
        data_section = _build_data_section(state)
        system_prompt = render_system_prompt(data=data_section, language_name=language_name)

        raw = _complete(provider, system_prompt, state["query"], tracer)

        def _regenerate(error_message: str) -> str:
            retry_prompt = system_prompt + render_retry_suffix(
                "generate_retry_invalid_json", error_message=error_message
            )
            return _complete(provider, retry_prompt, state["query"], tracer)

        parsed = validate_structured_output(raw, AgentResponse, regenerate=_regenerate)

        if not is_expected_script(parsed.answer, response_language):
            # One retry, same spirit as the JSON-validation retry above:
            # smaller open models occasionally answer hi requests in the
            # wrong script (verified on real production calls -- see
            # is_expected_script's docstring), and a retry sometimes
            # genuinely fixes it by chance rather than repeating the same
            # failure deterministically. (Kannada never reaches this branch
            # -- see the unconditional fallback above.)
            retry_prompt = system_prompt + render_retry_suffix(
                "generate_retry_wrong_script", language_name=language_name
            )
            try:
                retried_raw = _complete(provider, retry_prompt, state["query"], tracer)
                retried = validate_structured_output(retried_raw, AgentResponse, regenerate=_regenerate)
            except (ValidationError, LLMUnavailableError):
                retried = None
            if retried is not None and is_expected_script(retried.answer, response_language):
                parsed = retried
            else:
                fallback_answer = WRONG_SCRIPT_FALLBACK_ANSWER.get(response_language)
                if fallback_answer:
                    parsed = parsed.model_copy(update={"answer": fallback_answer})

    guardrail_flags = list(state.get("guardrail_flags", []))
    chunks = state.get("retrieved_chunks", [])
    if any(sanitize_with_flags(c.text)[1] for c in chunks):
        guardrail_flags.append("injection_marker_neutralised")

    failure_reason = None
    if state.get("route") == "policy" and response_language == "en" and chunks and not parsed.escalate_to_human:
        grounding = check_grounding(parsed.answer, [c.text for c in chunks], state.get("query", ""))
        if tracer:
            tracer.log_step(
                "grounding_check",
                decision="grounded" if grounding.grounded else f"UNGROUNDED: {grounding.reason}",
                word_overlap=round(grounding.word_overlap, 3),
                ungrounded_numbers=grounding.ungrounded_numbers,
            )
        if not grounding.grounded:
            guardrail_flags.append("ungrounded_answer")
            failure_reason = "ungrounded_answer"
            parsed = AgentResponse(
                answer=UNGROUNDED_ANSWER_MESSAGE, sources=[], confidence=0.0, escalate_to_human=True
            )

    # Classical override, not a request: don't trust the model's own
    # escalate_to_human when the classical sufficiency check already said the
    # context was insufficient. Verified necessary, not theoretical — a weaker
    # model (the HF fallback) confidently answered an off-corpus question
    # instead of escalating on one real run and correctly escalated on
    # another, purely by chance. Session 18's "add a verification step" theme,
    # applied the same way as the tool_error fix above: a classical check
    # beats hoping the model self-reports correctly.
    escalate_to_human = parsed.escalate_to_human or state.get("sufficient") is False

    # Disclaimer is fixed, code-appended text, never LLM-generated — same reason
    # production FinBuddy's low-confidence escalation string is fixed, not
    # LLM-translated: compliance-relevant wording must be guaranteed verbatim.
    disclaimer = (
        CREDIT_ASSESSMENT_DISCLAIMER.get(response_language, CREDIT_ASSESSMENT_DISCLAIMER["en"])
        if state.get("route") == "credit_assessment"
        else None
    )

    if tracer:
        # Logging the actual answer (not just a confidence number) is what
        # makes this a real audit trail of AI-assisted decisions, per Session
        # 18's "maintain a trail of AI assisted decisions" guardrail — a
        # confidence score alone doesn't let anyone reconstruct what was said.
        tracer.log_step(
            "generate",
            decision="produced structured response",
            answer=parsed.answer,
            confidence=parsed.confidence,
            escalate_to_human=escalate_to_human,
            model_self_reported_escalate=parsed.escalate_to_human,
            prompt_version=ACTIVE_VERSIONS["generate_system"],
        )

    return {
        **state,
        "answer": parsed.answer,
        # A fixed fallback answer (Kannada always; Hindi when generation failed twice) was NOT
        # produced from the retrieved text, so it must not cite it.
        "sources": [] if (failure_reason or parsed.answer in WRONG_SCRIPT_FALLBACK_ANSWER.values()) else _deterministic_sources(state),
        "confidence": parsed.confidence,
        "escalate_to_human": escalate_to_human,
        "disclaimer": disclaimer,
        "guardrail_flags": guardrail_flags,
        "failure_reason": failure_reason,
    }
