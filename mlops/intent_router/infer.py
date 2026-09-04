"""Load the production Intent Router and classify a message.

Falls back to a conservative keyword rule if no model has been promoted yet
(milestone step 5 not run) — so the graph is runnable before training exists.
"""
from __future__ import annotations

from pathlib import Path

import joblib

from mlops.intent_router.data import ROUTES
from mlops.intent_router.registry import get_current_production_artifact

_FALLBACK_KEYWORDS = {
    "credit_assessment": ("score", "credit", "approved", "loan", "risk trend", "borrow"),
    "policy": ("data", "privacy", "consent", "rbi", "dpdp", "retention", "limit"),
}

_model = None


def _load_model():
    global _model
    if _model is None:
        artifact_path = get_current_production_artifact()
        if artifact_path is not None:
            _model = joblib.load(artifact_path)
    return _model


def _keyword_fallback(message: str) -> tuple[str, float]:
    lowered = message.lower()
    for route, keywords in _FALLBACK_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return route, 0.5  # low, honest confidence — this is a fallback, not a trained model
    return "off_topic", 0.5


def classify_intent(message: str) -> tuple[str, float]:
    """Returns (route, confidence). route is one of mlops.intent_router.data.ROUTES."""
    model = _load_model()
    if model is None:
        return _keyword_fallback(message)

    proba = model.predict_proba([message])[0]
    classes = model.classes_
    best_idx = proba.argmax()
    route = classes[best_idx]
    confidence = float(proba[best_idx])
    assert route in ROUTES
    return route, confidence
