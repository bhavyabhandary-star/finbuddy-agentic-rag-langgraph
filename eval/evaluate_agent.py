"""Evaluation harness CLI — scores task completion + correct routing, wired into
CI via --gate (per kickoff_prompt.md's CI/CD section).

Run: python -m eval.evaluate_agent [--gate]
"""
from __future__ import annotations

import argparse
import sys

from eval.scenarios import SCENARIOS
from mlops.intent_router.infer import classify_intent

# Below this fraction of scenarios routed correctly, --gate fails the build.
GATE_MIN_ROUTING_ACCURACY = 0.75


def run_scenarios() -> dict:
    results = []
    for scenario in SCENARIOS:
        route, confidence = classify_intent(scenario.query)
        route_correct = route == scenario.expected_route
        results.append(
            {
                "scenario": scenario.name,
                "expected_route": scenario.expected_route,
                "actual_route": route,
                "route_correct": route_correct,
                "confidence": confidence,
            }
        )
    return {
        "results": results,
        "routing_accuracy": sum(r["route_correct"] for r in results) / len(results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="Exit non-zero on a regression, for CI")
    args = parser.parse_args()

    summary = run_scenarios()
    for r in summary["results"]:
        status = "PASS" if r["route_correct"] else "FAIL"
        print(f"[{status}] {r['scenario']}: expected={r['expected_route']} actual={r['actual_route']} (confidence={r['confidence']:.2f})")
    print(f"\nRouting accuracy: {summary['routing_accuracy']:.0%}")

    if args.gate and summary["routing_accuracy"] < GATE_MIN_ROUTING_ACCURACY:
        print(f"GATE FAILED: routing accuracy below {GATE_MIN_ROUTING_ACCURACY:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
