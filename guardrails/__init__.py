"""Guardrail Layer (build_prompt.md / kickoff_prompt.md, Layer 5).

input_guardrails  — sanitize retrieved PDF text, flag PII before it reaches an LLM
tool_policy       — safe_tool_call wrapper (timeout/retry/error-as-observation),
                     confirmation-gate framework for any future side-effecting tool
output_guardrails — Pydantic validation with retry-with-error-fed-back
"""
