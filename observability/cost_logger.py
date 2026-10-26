"""Tracks steps, LLM calls, tokens, latency and cost per run -- per kickoff_prompt.md's
"Numbers to Know" note: be ready to quote these, don't guess them at demo time.

Wired into every LLM call (agent/nodes/generate.py::_complete) and written to the run's
trace as a final `run_summary` line. Two honesty rules:

* Tokens are the provider's own usage numbers when it reports them; otherwise a labelled
  ~4-characters-per-token ESTIMATE (`tokens_estimated=True`), never presented as measured.
* Cost needs a price. The old placeholder was a Claude price list applied to a Qwen model.
  Now the price comes from LLM_PRICE_INPUT_PER_1K / LLM_PRICE_OUTPUT_PER_1K (config/settings.py),
  and with none set `estimated_cost_usd` is None -- "unknown", not a made-up number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import settings

_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


@dataclass
class RunCostTracker:
    steps: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_seconds: float = 0.0
    tokens_estimated: bool = False
    tool_calls: dict = field(default_factory=dict)

    def record_step(self) -> None:
        self.steps += 1

    def record_llm_call(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        *,
        prompt_text: str = "",
        completion_text: str = "",
        reported_usage: tuple[int, int] | None = None,
        seconds: float = 0.0,
    ) -> None:
        if reported_usage is not None:
            input_tokens, output_tokens = reported_usage
        if input_tokens is None or output_tokens is None:
            input_tokens, output_tokens = estimate_tokens(prompt_text), estimate_tokens(completion_text)
            self.tokens_estimated = True
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.llm_seconds += seconds

    def record_tool_call(self, tool_name: str) -> None:
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1

    @property
    def estimated_cost_usd(self) -> float | None:
        price = settings.llm_price_per_1k_tokens()
        if price is None:
            return None
        per_in, per_out = price
        return self.input_tokens / 1000 * per_in + self.output_tokens / 1000 * per_out

    def summary(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_estimated": self.tokens_estimated,
            "llm_seconds": round(self.llm_seconds, 3),
            "estimated_cost_usd": self.estimated_cost_usd,
        }
