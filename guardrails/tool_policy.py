"""Tool-call policy: safe_tool_call wrapper + confirmation-gate framework.

Per kickoff_prompt.md: catch tool errors explicitly and feed them back as an
observation, not a crash; cap retries; define a fallback. The confirmation-gate
function exists because kickoff_prompt.md marks it non-negotiable for any tool
with a real-world side effect — this MVP's tools are all read-only, so the gate
is defined but not yet triggered; document that honestly rather than skip it.
"""
from __future__ import annotations

import logging
from typing import Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger("guardrails.tool_policy")

MAX_RETRIES_PER_TOOL_CALL = 1  # cap per kickoff_prompt.md's cost/latency section


def safe_tool_call(
    fn: Callable[[], T],
    *,
    tool_name: str,
    fallback: Callable[[Exception], T],
    max_retries: int = MAX_RETRIES_PER_TOOL_CALL,
) -> T:
    """Runs fn(), retrying up to max_retries times, falling back on final failure.

    The fallback result must carry enough signal (e.g. a `tool_error` field) that
    the agent can decide to degrade gracefully or escalate — never fail silently.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — deliberately broad: any tool failure is an observation
            last_exc = exc
            logger.warning("tool_call_failed", extra={"tool": tool_name, "attempt": attempt, "error": str(exc)})
    assert last_exc is not None
    return fallback(last_exc)


class SideEffectNotConfirmedError(Exception):
    """Raised by require_confirmation() when a side-effecting tool is called without one."""


def require_confirmation(tool_name: str, user_confirmed: bool) -> None:
    """Non-negotiable per kickoff_prompt.md: any tool with a real-world side effect
    (sending a message, writing to a DB) needs an explicit confirmation gate.

    No tool in this MVP has a side effect yet (retrieval and read-only scoring calls
    don't need one) — this function exists so the pattern is in place and testable
    before it's ever actually load-bearing, not bolted on after the fact.
    """
    if not user_confirmed:
        raise SideEffectNotConfirmedError(
            f"'{tool_name}' has a real-world side effect and was not confirmed."
        )
