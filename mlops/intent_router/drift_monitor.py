"""Drift monitoring for the Intent Router.

Two independent signals, because there is no live ground truth for "was that route right":

1. MISROUTE RATE -- the only direct measure. Fed by human feedback (POST /agent/feedback
   with route_was_correct / correct_route). Until now record_routing_outcome() was defined
   but never called from anywhere, so this always read 0.0 and the tier was green forever.
2. ROUTING-CONFIDENCE DRIFT -- an early-warning proxy that needs no labels: the share of
   live queries the router is unsure about (confidence < LOW_ROUTE_CONFIDENCE). If users
   start asking things the router was not trained on, that share climbs before anyone
   files a correction. Thresholds are initial, calibrated on the held-out set, not on
   live traffic (there is none yet) -- see config/settings.py.

Full ownership per build_prompt.md -- this is the one model this project trains, so it's
the one model this project also retrains on a breach.
"""
from __future__ import annotations

from collections import deque

from config.settings import (
    DRIFT_MIN_SAMPLES,
    MISROUTE_MIN_FEEDBACK,
    LOW_CONFIDENCE_RATE_AMBER,
    LOW_CONFIDENCE_RATE_RED,
    LOW_ROUTE_CONFIDENCE,
)

AMBER_MISROUTE_RATE = 0.10
RED_MISROUTE_RATE = 0.20

_recent_outcomes: deque[bool] = deque(maxlen=200)  # True = correct route
_recent_confidences: deque[float] = deque(maxlen=200)


def record_routing_outcome(was_correct: bool) -> None:
    _recent_outcomes.append(was_correct)


def record_route_confidence(confidence: float) -> None:
    _recent_confidences.append(confidence)


def reset() -> None:
    _recent_outcomes.clear()
    _recent_confidences.clear()


def current_misroute_rate() -> float:
    if not _recent_outcomes:
        return 0.0
    return 1.0 - (sum(_recent_outcomes) / len(_recent_outcomes))


def low_confidence_rate() -> float:
    if not _recent_confidences:
        return 0.0
    return sum(c < LOW_ROUTE_CONFIDENCE for c in _recent_confidences) / len(_recent_confidences)


def drift_tier() -> str:
    """Returns "green" | "amber" | "red" | "insufficient_data" per build_prompt.md's drift
    response ladder (misroute rate from human feedback)."""
    if len(_recent_outcomes) < MISROUTE_MIN_FEEDBACK:
        return "insufficient_data"  # one thumbs-down is not a drift signal
    rate = current_misroute_rate()
    if rate >= RED_MISROUTE_RATE:
        return "red"
    if rate >= AMBER_MISROUTE_RATE:
        return "amber"
    return "green"


def confidence_drift_tier() -> str:
    """"green" | "amber" | "red" | "insufficient_data" (fewer than DRIFT_MIN_SAMPLES requests)."""
    if len(_recent_confidences) < DRIFT_MIN_SAMPLES:
        return "insufficient_data"
    rate = low_confidence_rate()
    if rate >= LOW_CONFIDENCE_RATE_RED:
        return "red"
    if rate >= LOW_CONFIDENCE_RATE_AMBER:
        return "amber"
    return "green"


def drift_summary() -> dict:
    return {
        "misroute_tier": drift_tier(),
        "misroute_rate": round(current_misroute_rate(), 4),
        "feedback_samples": len(_recent_outcomes),
        "confidence_tier": confidence_drift_tier(),
        "low_confidence_rate": round(low_confidence_rate(), 4),
        "confidence_samples": len(_recent_confidences),
    }
