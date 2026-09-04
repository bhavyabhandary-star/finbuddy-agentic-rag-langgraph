"""Tracks steps-per-run and cost-per-run — per kickoff_prompt.md's "Numbers to
Know" note: be ready to quote these, don't guess them at demo time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Rough Claude pricing placeholder — replace with the exact model's published
# per-token rate before quoting a real cost-per-run number anywhere.
USD_PER_1K_INPUT_TOKENS = 0.003
USD_PER_1K_OUTPUT_TOKENS = 0.015


@dataclass
class RunCostTracker:
    steps: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: dict = field(default_factory=dict)

    def record_step(self) -> None:
        self.steps += 1

    def record_llm_call(self, input_tokens: int, output_tokens: int) -> None:
        self.llm_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def record_tool_call(self, tool_name: str) -> None:
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens / 1000 * USD_PER_1K_INPUT_TOKENS
            + self.output_tokens / 1000 * USD_PER_1K_OUTPUT_TOKENS
        )
