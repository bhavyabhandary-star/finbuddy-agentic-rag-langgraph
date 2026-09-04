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


class ClaudeProvider(LLMProvider):
    def complete(self, system_prompt: str, user_query: str, max_tokens: int) -> str:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-3-5-sonnet-latest", max_tokens=max_tokens)
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

    if tracer:
        tracer.log_step("generate", decision="produced structured response", confidence=parsed.confidence)

    return {
        **state,
        "answer": parsed.answer,
        "sources": parsed.sources,
        "confidence": parsed.confidence,
        "escalate_to_human": parsed.escalate_to_human,
    }
