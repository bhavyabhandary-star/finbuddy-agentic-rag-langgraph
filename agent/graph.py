"""The LangGraph state machine — wires route/retrieve/credit_assessment/generate
into the agentic loop from kickoff_prompt.md.

    Query -> Router -> [credit_assessment -> Generate]
                     -> [Retrieve -> Sufficiency-Check -(insufficient)-> Retrieve]
                                                        -(sufficient)--> Generate
                     -> [off_topic -> escalate, no LLM call]
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.nodes.credit_assessment import credit_assessment_node
from agent.nodes.generate import generate_node
from agent.nodes.retrieve import retrieve_node
from agent.nodes.route import route_node
from agent.state import MAX_LOOP_ITERATIONS, AgentState
from observability.tracing import RunTracer


def _escalate_node(state: AgentState, tracer: RunTracer | None = None) -> AgentState:
    """Off-topic path: no LLM call at all — cheaper and faster, exactly as
    kickoff_prompt.md's production-FinBuddy example demonstrates (2s vs 9s).
    """
    if tracer:
        tracer.log_step("escalate", decision="off-topic query, routed to human without an LLM call")
    return {
        **state,
        "answer": "I don't have a confident, verified answer to that. Let me connect you with a human coach.",
        "sources": [],
        "confidence": 0.0,
        "escalate_to_human": True,
    }


def _route_decision(state: AgentState) -> str:
    return {"policy": "retrieve", "credit_assessment": "credit_assessment", "off_topic": "escalate"}[
        state["route"]
    ]


def _should_continue_retrieval(state: AgentState) -> str:
    if state.get("sufficient"):
        return "generate"
    if state.get("loop_count", 0) >= MAX_LOOP_ITERATIONS:
        # Cap hit without sufficiency — escalate rather than loop forever or
        # generate ungrounded (per kickoff_prompt.md's guardrail table).
        return "escalate"
    return "retrieve"


def build_graph(tracer: RunTracer | None = None):
    graph = StateGraph(AgentState)

    graph.add_node("route", lambda s: route_node(s, tracer))
    graph.add_node("retrieve", lambda s: retrieve_node(s, tracer))
    graph.add_node(
        "credit_assessment",
        lambda s: credit_assessment_node(
            s, signals=s.get("credit_signals", {}), delta_features=s.get("risk_trend_delta_features"), tracer=tracer
        ),
    )
    graph.add_node("generate", lambda s: generate_node(s, tracer=tracer))
    graph.add_node("escalate", lambda s: _escalate_node(s, tracer))

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route", _route_decision, {"retrieve": "retrieve", "credit_assessment": "credit_assessment", "escalate": "escalate"}
    )
    graph.add_conditional_edges(
        "retrieve", _should_continue_retrieval, {"retrieve": "retrieve", "generate": "generate", "escalate": "escalate"}
    )
    graph.add_edge("credit_assessment", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("escalate", END)

    return graph.compile()
