"""Router node: classical classification, NOT an LLM call (constraint 1).

Uses the Intent Router (mlops/intent_router) — the one model this project owns —
to decide policy / credit_assessment / off_topic before any tool runs.
"""
from __future__ import annotations

from agent.state import AgentState
from mlops.intent_router.infer import classify_intent
from observability.tracing import RunTracer


def route_node(state: AgentState, tracer: RunTracer | None = None) -> AgentState:
    route, confidence = classify_intent(state["query"])
    if tracer:
        tracer.log_step(
            "route",
            decision=f"classified as '{route}'",
            confidence=confidence,
            query=state["query"],
        )
    return {**state, "route": route, "route_confidence": confidence, "loop_count": 0}
