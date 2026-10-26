"""The eval must not be able to grade the router on its own homework again.

Background: eval/scenarios.py's four queries are word-for-word members of the router's
training set, so "100% on the eval harness" measured memorisation, and CI's gate was a
train-on-test check. These tests make that failure mode loud.
"""
from __future__ import annotations

import re

import pytest

from eval.blind_scenarios import BLIND_SCENARIOS
from eval.evaluate_agent import gate_failures
from eval.heldout_scenarios import HELDOUT_SCENARIOS
from eval.scenarios import SCENARIOS
from mlops.intent_router.data import ALL_EXAMPLES, ROUTES, load_training_examples

VALID_TAGS = {"plain", "paraphrase", "hinglish", "typo", "ambiguous", "adversarial", "pii"}
NEAR_DUPLICATE_JACCARD = 0.7


def _tokens(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", "", text.lower()).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def _training_texts() -> list[str]:
    # include_feedback=True: reviewed user corrections are training data too, and must
    # never contain an eval query either.
    texts, _ = load_training_examples(include_feedback=True)
    return texts


@pytest.mark.parametrize("name,scenarios", [("heldout", HELDOUT_SCENARIOS), ("blind", BLIND_SCENARIOS)])
def test_eval_sets_never_overlap_training_data(name, scenarios):
    train_tokens = [_tokens(t) for t in _training_texts()]
    train_exact = {t.strip().lower() for t in _training_texts()}
    for s in scenarios:
        assert s.query.strip().lower() not in train_exact, f"{name} query is in the training set: {s.query!r}"
        closest = max(_jaccard(_tokens(s.query), t) for t in train_tokens)
        assert closest < NEAR_DUPLICATE_JACCARD, (
            f"{name} query is a near-duplicate of a training example (jaccard {closest:.2f}): {s.query!r}"
        )


def test_heldout_and_blind_sets_do_not_overlap_each_other():
    heldout = {s.query.strip().lower() for s in HELDOUT_SCENARIOS}
    assert not heldout & {s.query.strip().lower() for s in BLIND_SCENARIOS}


@pytest.mark.parametrize("scenarios", [HELDOUT_SCENARIOS, BLIND_SCENARIOS])
def test_eval_sets_are_well_formed_and_cover_every_route_and_edge_case(scenarios):
    assert all(s.expected_route in ROUTES for s in scenarios)
    assert all(s.tag in VALID_TAGS for s in scenarios)
    for route in ROUTES:
        assert sum(s.expected_route == route for s in scenarios) >= 10, f"too few '{route}' queries"
    assert {"typo", "hinglish", "adversarial", "pii"} <= {s.tag for s in scenarios}


def test_legacy_smoke_scenarios_are_known_to_be_in_the_training_set():
    """Pins the old harness's flaw as a fact rather than letting it be forgotten: these four
    ARE training data, so they may smoke-test the pipeline but can never be an accuracy claim."""
    training = {t.strip().lower() for t, _ in ALL_EXAMPLES}
    assert all(s.query.strip().lower() in training for s in SCENARIOS)


def test_gate_fails_when_a_whole_route_collapses_even_if_average_accuracy_is_fine():
    report = {
        "accuracy": 0.80,
        "per_route": {
            "policy": {"n": 20, "recall": 1.0, "precision": 0.8},
            "credit_assessment": {"n": 20, "recall": 0.95, "precision": 0.9},
            "off_topic": {"n": 20, "recall": 0.4, "precision": 1.0},
        },
    }
    failures = gate_failures(report, {"dev-set": 0.9, "blind-set": 0.8})
    assert any("off_topic" in f for f in failures)


def test_gate_fails_when_the_blind_set_is_weak_even_if_the_dev_set_is_strong():
    report = {"accuracy": 0.9, "per_route": {r: {"n": 10, "recall": 0.9, "precision": 0.9} for r in ROUTES}}
    failures = gate_failures(report, {"dev-set": 0.97, "blind-set": 0.6})
    assert any("blind-set" in f for f in failures)


def test_gate_passes_a_healthy_report():
    report = {"accuracy": 0.9, "per_route": {r: {"n": 10, "recall": 0.9, "precision": 0.9} for r in ROUTES}}
    assert gate_failures(report, {"dev-set": 0.95, "blind-set": 0.85}) == []
