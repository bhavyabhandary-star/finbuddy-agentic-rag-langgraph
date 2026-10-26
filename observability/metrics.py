"""Live service metrics, exposed at GET /metrics.

In-memory and per-process: they reset on restart and are not aggregated across workers --
the honest scope of an MVP with no metrics backend. They exist so that "is it healthy, and
is the router still seeing the kind of traffic it was trained for" is answerable from the
running service instead of by reading trace files by hand.

What it watches:
* volume, escalation rate, and the breakdown of WHY answers were fixed fallbacks
  (llm_unavailable / invalid_structured_output / ungrounded_answer / internal_error)
* latency p50 / p95 over the last LATENCY_WINDOW requests, cache hit rate
* LLM calls, tokens and (when a price is configured) cost
* router health: confidence distribution drift, plus the misroute rate fed by user feedback
"""
from __future__ import annotations

import threading
from collections import Counter, deque

from config import settings
from mlops.intent_router import drift_monitor


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[index], 3)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.requests = 0
            self.escalations = 0
            self.cache_hits = 0
            self.pii_redacted_requests = 0
            self.by_route: Counter[str] = Counter()
            self.failures: Counter[str] = Counter()
            self.llm_calls = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.cost_usd = 0.0
            self.cost_known = True
            self.latencies: deque[float] = deque(maxlen=settings.LATENCY_WINDOW)

    def record_request(
        self,
        *,
        route: str | None,
        route_confidence: float | None,
        escalated: bool,
        failure_reason: str | None,
        cache_hit: bool,
        latency_seconds: float,
        pii_redacted: bool,
        cost_summary: dict | None = None,
    ) -> None:
        with self._lock:
            self.requests += 1
            self.escalations += int(escalated)
            self.cache_hits += int(cache_hit)
            self.pii_redacted_requests += int(pii_redacted)
            if route:
                self.by_route[route] += 1
            if failure_reason:
                self.failures[failure_reason] += 1
            self.latencies.append(latency_seconds)
            if cost_summary:
                self.llm_calls += cost_summary.get("llm_calls", 0)
                self.input_tokens += cost_summary.get("input_tokens", 0)
                self.output_tokens += cost_summary.get("output_tokens", 0)
                if cost_summary.get("estimated_cost_usd") is None:
                    self.cost_known = False if cost_summary.get("llm_calls") else self.cost_known
                else:
                    self.cost_usd += cost_summary["estimated_cost_usd"]
        if route_confidence is not None and not cache_hit:
            drift_monitor.record_route_confidence(route_confidence)

    def snapshot(self) -> dict:
        with self._lock:
            n = self.requests
            latencies = list(self.latencies)
            snap = {
                "requests": n,
                "escalation_rate": round(self.escalations / n, 4) if n else None,
                "cache_hit_rate": round(self.cache_hits / n, 4) if n else None,
                "pii_redacted_requests": self.pii_redacted_requests,
                "by_route": dict(self.by_route),
                "fixed_fallbacks_by_reason": dict(self.failures),
                "latency_seconds": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95), "window": len(latencies)},
                "llm": {
                    "calls": self.llm_calls,
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "estimated_cost_usd": round(self.cost_usd, 6) if self.cost_known and self.llm_calls else None,
                    "note": "cost is null unless LLM_PRICE_INPUT_PER_1K / LLM_PRICE_OUTPUT_PER_1K are set",
                },
            }
        snap["router_drift"] = drift_monitor.drift_summary()
        return snap


metrics = Metrics()
