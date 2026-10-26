"""Evaluation harness CLI — scores correct routing, wired into CI via --gate (per
kickoff_prompt.md's CI/CD section).

What it measures, and what it deliberately no longer does:

* The GATE scores queries the router has never trained on (eval/heldout_scenarios.py,
  plus eval/blind_scenarios.py written after the last round of fixes). It used to score the
  four scenarios in eval/scenarios.py, which are word-for-word members of the router's
  training set -- that reported 100% regardless of how good the router was, and CI's gate
  was therefore testing memorisation. Those four remain only as a pipeline SMOKE check
  (does the graph still route and call the right tools?), printed separately and never
  counted as accuracy.
* Accuracy is not enough: a router that never says "off_topic" can still average 75%. The
  gate also fails if any ONE route's recall drops below GATE_MIN_PER_ROUTE_RECALL, and the
  report shows the confusion matrix and the accuracy on each edge-case tag (typo, Hinglish,
  adversarial, PII, ambiguous), with every miss listed.

Run: python -m eval.evaluate_agent [--gate] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

from config.settings import GATE_MIN_PER_ROUTE_RECALL, GATE_MIN_ROUTING_ACCURACY
from eval.blind_scenarios import BLIND_SCENARIOS
from eval.heldout_scenarios import HELDOUT_SCENARIOS
from eval.scenarios import SCENARIOS
from mlops.intent_router.data import ROUTES
from mlops.intent_router.infer import classify_intent


def run_scenarios() -> dict:
    """Pipeline smoke check on the 4 original scenarios. NOT an accuracy measure: they are
    in the router's training set (tests/test_eval_integrity.py pins that this stays visible)."""
    results = []
    for scenario in SCENARIOS:
        route, confidence = classify_intent(scenario.query)
        results.append(
            {
                "scenario": scenario.name,
                "expected_route": scenario.expected_route,
                "actual_route": route,
                "route_correct": route == scenario.expected_route,
                "confidence": confidence,
            }
        )
    return {"results": results, "routing_accuracy": sum(r["route_correct"] for r in results) / len(results)}


def run_heldout(scenarios=None) -> dict:
    scenarios = list(HELDOUT_SCENARIOS) + list(BLIND_SCENARIOS) if scenarios is None else scenarios
    rows = []
    for s in scenarios:
        route, confidence = classify_intent(s.query)
        rows.append(
            {
                "query": s.query,
                "tag": s.tag,
                "expected": s.expected_route,
                "actual": route,
                "correct": route == s.expected_route,
                "confidence": round(confidence, 3),
            }
        )

    n = len(rows)
    per_route: dict[str, dict] = {}
    for route in ROUTES:
        expected = [r for r in rows if r["expected"] == route]
        predicted = [r for r in rows if r["actual"] == route]
        tp = sum(r["correct"] for r in expected)
        per_route[route] = {
            "n": len(expected),
            "recall": tp / len(expected) if expected else None,
            "precision": (sum(r["expected"] == route for r in predicted) / len(predicted)) if predicted else None,
        }
    by_tag: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r["correct"])
    confusion = Counter((r["expected"], r["actual"]) for r in rows if not r["correct"])
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "per_route": per_route,
        "by_tag": {t: {"n": len(v), "accuracy": sum(v) / len(v)} for t, v in by_tag.items()},
        "confusion_misses": {f"{e} -> {a}": c for (e, a), c in confusion.most_common()},
        "misses": [r for r in rows if not r["correct"]],
        "mean_confidence_correct": _mean([r["confidence"] for r in rows if r["correct"]]),
        "mean_confidence_wrong": _mean([r["confidence"] for r in rows if not r["correct"]]),
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def gate_failures(report: dict, per_set_accuracy: dict[str, float] | None = None) -> list[str]:
    """Fails if EITHER set is below the accuracy bar (so a strong dev-set score cannot hide a
    weak blind-set one), or if any route's recall on the combined set is below the floor."""
    failures = []
    for name, accuracy in (per_set_accuracy or {"held-out": report["accuracy"]}).items():
        if accuracy < GATE_MIN_ROUTING_ACCURACY:
            failures.append(f"{name} routing accuracy {accuracy:.0%} < {GATE_MIN_ROUTING_ACCURACY:.0%}")
    for route, stats in report["per_route"].items():
        if stats["recall"] is not None and stats["recall"] < GATE_MIN_PER_ROUTE_RECALL:
            failures.append(f"recall for '{route}' is {stats['recall']:.0%} < {GATE_MIN_PER_ROUTE_RECALL:.0%}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="Exit non-zero on a regression, for CI")
    parser.add_argument("--json", metavar="PATH", help="Also write the full report as JSON")
    args = parser.parse_args()

    smoke = run_scenarios()
    print("Pipeline smoke (4 scenarios that ARE in the training set -- not an accuracy measure):")
    for r in smoke["results"]:
        status = "PASS" if r["route_correct"] else "FAIL"
        print(f"  [{status}] {r['scenario']}: expected={r['expected_route']} actual={r['actual_route']} (confidence={r['confidence']:.2f})")

    dev, blind = run_heldout(list(HELDOUT_SCENARIOS)), run_heldout(list(BLIND_SCENARIOS))
    report = run_heldout()  # combined
    print("\nHeld-out routing -- queries the router never trained on")
    print(f"  dev set   (eval/heldout_scenarios.py, used to diagnose + harden): {dev['accuracy']:.1%}  (n={dev['n']}; optimistic after hardening)")
    print(f"  blind set (eval/blind_scenarios.py, written after hardening)    : {blind['accuracy']:.1%}  (n={blind['n']}; the number to quote)")
    print(f"  combined: {report['accuracy']:.1%} (n={report['n']})")
    for route, st in report["per_route"].items():
        recall = "n/a" if st["recall"] is None else f"{st['recall']:.0%}"
        precision = "n/a" if st["precision"] is None else f"{st['precision']:.0%}"
        print(f"  {route:<18} n={st['n']:<3} recall={recall:<5} precision={precision}")
    print("  by tag: " + ", ".join(f"{t} {v['accuracy']:.0%} (n={v['n']})" for t, v in sorted(report["by_tag"].items())))
    print(f"  mean confidence  when right: {report['mean_confidence_correct']}  when wrong: {report['mean_confidence_wrong']}")
    if report["misses"]:
        print("  misses:")
        for m in report["misses"]:
            print(f"    [{m['tag']}] expected={m['expected']} got={m['actual']} ({m['confidence']}) :: {m['query']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"smoke": smoke, "dev": dev, "blind": blind, "combined": report}, f, indent=2)

    failures = gate_failures(report, {"dev-set": dev["accuracy"], "blind-set": blind["accuracy"]})
    if args.gate and failures:
        for failure in failures:
            print(f"GATE FAILED: {failure}")
        return 1
    if args.gate:
        print("\nGate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
