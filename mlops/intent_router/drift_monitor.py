"""Drift monitoring for the Intent Router: routing-confidence distribution + a
misroute rate tracked from eval-harness scenarios or user corrections.

Full ownership per build_prompt.md — this is the one model this project trains,
so it's the one model this project also retrains on a breach.
"""
from __future__ import annotations

from collections import deque

AMBER_MISROUTE_RATE = 0.10
RED_MISROUTE_RATE = 0.20

_recent_outcomes: deque[bool] = deque(maxlen=200)  # True = correct route


def record_routing_outcome(was_correct: bool) -> None:
    _recent_outcomes.append(was_correct)


def current_misroute_rate() -> float:
    if not _recent_outcomes:
        return 0.0
    return 1.0 - (sum(_recent_outcomes) / len(_recent_outcomes))


def drift_tier() -> str:
    """Returns "green" | "amber" | "red" per build_prompt.md's drift response ladder."""
    rate = current_misroute_rate()
    if rate >= RED_MISROUTE_RATE:
        return "red"
    if rate >= AMBER_MISROUTE_RATE:
        return "amber"
    return "green"
