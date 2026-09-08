"""Generate node — the ONLY mandatory LLM call in the whole graph.

Structured Role/Instructions/Constraints/Data/Output-format prompt (Session 14,
slide 31), forced Pydantic-validated JSON, max_tokens set, provider-agnostic
(Claude default, Ollama fallback verified once — see kickoff_prompt.md).
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from agent.state import AgentState
from guardrails.input_guardrails import sanitize_retrieved_text
from guardrails.output_guardrails import validate_structured_output
from observability.tracing import RunTracer
from tools.schemas import AgentResponse

MAX_OUTPUT_TOKENS = 512  # explicit cap per kickoff_prompt.md's cost/latency section

SYSTEM_PROMPT_TEMPLATE = """\
## ROLE:
You are FinBuddy, an empathetic credit coach for gig-economy workers in India.
You are not a bank — you are a trusted financial coach.

## INSTRUCTIONS:
Answer the user's question using ONLY the DATA provided below. If the data is
insufficient to answer confidently, set escalate_to_human to true instead of
guessing.

## CONSTRAINTS:
- Never invent a policy detail not present in DATA.
- Never use protected attributes (gender, religion, caste, pincode) as reasoning.
- Keep the answer under 4 sentences, plain language, no jargon.

## DATA:
{data}

## OUTPUT FORMAT (strict JSON, matching this schema exactly):
{{"answer": "...", "sources": ["..."], "confidence": 0.0, "escalate_to_human": false}}
"""


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        """Returns raw completion text (expected to be JSON per the schema)."""


DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"


class ClaudeProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL):
        self.model = model

    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=self.model, max_tokens=max_tokens)
        response = llm.invoke([("system", system_prompt), ("human", user_query)])
        return response.content


class OllamaProvider(LLMProvider):
    """Local-model fallback — verify once with a real run, keep the proof, but
    this is not the default demo path (see build_prompt.md's resolved decisions).
    """

    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        import ollama

        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            options={"num_predict": max_tokens},
        )
        return response["message"]["content"]


DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # ungated, broadly available across HF Inference Providers


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
        self.model = model or os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
        self.provider = provider or os.environ.get("HF_PROVIDER", "featherless-ai")

    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        from huggingface_hub import InferenceClient

        client = InferenceClient(
            model=self.model, provider=self.provider, token=os.environ.get("HF_TOKEN")
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
        return response.choices[0].message.content


CREDIT_ASSESSMENT_DISCLAIMER = (
    "This is an AI-generated explanation of an automated assessment, not "
    "financial advice. Treat it as decision-support — confirm any important "
    "decision with FinBuddy support."
)


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


def generate_node(
    state: AgentState, provider: LLMProvider | None = None, tracer: RunTracer | None = None
) -> AgentState:
    provider = provider or ClaudeProvider()
    data_section = _build_data_section(state)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(data=data_section)

    raw = provider.complete(system_prompt, state["query"], MAX_OUTPUT_TOKENS)

    def _regenerate(error_message: str) -> str:
        retry_prompt = system_prompt + f"\n\nYour previous output was invalid: {error_message}\nReturn ONLY valid JSON."
        return provider.complete(retry_prompt, state["query"], MAX_OUTPUT_TOKENS)

    parsed: AgentResponse = validate_structured_output(raw, AgentResponse, regenerate=_regenerate)

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
    disclaimer = CREDIT_ASSESSMENT_DISCLAIMER if state.get("route") == "credit_assessment" else None

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
        )

    return {
        **state,
        "answer": parsed.answer,
        "sources": parsed.sources,
        "confidence": parsed.confidence,
        "escalate_to_human": escalate_to_human,
        "disclaimer": disclaimer,
    }
