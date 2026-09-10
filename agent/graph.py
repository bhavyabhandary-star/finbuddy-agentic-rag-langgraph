"""The LangGraph state machine — wires route/retrieve/credit_assessment/generate
into the agentic loop from kickoff_prompt.md.

    Query -> Router -> [credit_assessment -(tool succeeded)-> Generate
                                           -(tool_error)-----> escalate]
                     -> [Retrieve -> Sufficiency-Check -(insufficient)-> Retrieve]
                                                        -(sufficient)--> Generate
                     -> [off_topic -> escalate, no LLM call]
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.nodes.credit_assessment import credit_assessment_node
from agent.nodes.generate import LLMProvider, generate_node
from agent.nodes.retrieve import retrieve_node
from agent.nodes.route import route_node
from agent.state import MAX_LOOP_ITERATIONS, AgentState
from observability.tracing import RunTracer


_DEFAULT_ESCALATION_MESSAGE = (
    "I don't have a confident, verified answer to that. Let me connect you with a human coach."
)
_TOOL_ERROR_ESCALATION_MESSAGE = (
    "I'm unable to reach FinBuddy's scoring system right now, so I can't give you a "
    "verified assessment. Please try again shortly, or a human coach can help directly."
)


def _escalate_node(state: AgentState, tracer: RunTracer | None = None) -> AgentState:
    """No LLM call at all on this path — cheaper and faster, exactly as
    kickoff_prompt.md's production-FinBuddy example demonstrates (2s vs 9s), and
    it means a failed tool call is never narrated by an LLM as if it were data.
    """
    credit_assessment = state.get("credit_assessment")
    if credit_assessment is not None and credit_assessment.tool_error:
        message = _TOOL_ERROR_ESCALATION_MESSAGE
        reason = f"credit-assessment tool failed: {credit_assessment.tool_error}"
    else:
        message = _DEFAULT_ESCALATION_MESSAGE
        reason = "off-topic or insufficiently-grounded query"

    if tracer:
        tracer.log_step("escalate", decision=reason, routed_to_human_without_llm_call=True)

    return {
        **state,
        "answer": message,
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


def _should_generate_from_credit_assessment(state: AgentState) -> str:
    """Bug found via Session 18's "wrong information" / decision-support-not-
    autonomous-decision-maker guardrail theme: without this check, a failed
    assess_credit_profile call still had credit_assessment_node mark the state
    "sufficient" and flow straight to Generate — which would then explain the
    tool's fallback zero-data (credit_score=0, approved=False) as if it were a
    real assessment. A tool failure must escalate, not be narrated as a result.
    """
    credit_assessment = state.get("credit_assessment")
    if credit_assessment is not None and credit_assessment.tool_error:
        return "escalate"
    return "generate"


def build_graph(tracer: RunTracer | None = None, provider: LLMProvider | None = None):
    """provider is injectable (defaults to Claude inside generate_node when
    None) so callers can substitute HuggingFaceProvider/OllamaProvider without
    monkeypatching — needed for real end-to-end verification while Claude API
    billing is unresolved, and matches kickoff_prompt.md's LLM-agnostic design.
    """
    graph = StateGraph(AgentState)

    graph.add_node("route", lambda s: route_node(s, tracer))
    graph.add_node("retrieve", lambda s: retrieve_node(s, tracer))
    graph.add_node(
        "credit_assessment",
        lambda s: credit_assessment_node(
            s, signals=s.get("credit_signals", {}), delta_features=s.get("risk_trend_delta_features"), tracer=tracer
        ),
    )
    graph.add_node("generate", lambda s: generate_node(s, provider=provider, tracer=tracer))
    graph.add_node("escalate", lambda s: _escalate_node(s, tracer))

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route", _route_decision, {"retrieve": "retrieve", "credit_assessment": "credit_assessment", "escalate": "escalate"}
    )
    graph.add_conditional_edges(
        "retrieve", _should_continue_retrieval, {"retrieve": "retrieve", "generate": "generate", "escalate": "escalate"}
    )
    graph.add_conditional_edges(
        "credit_assessment", _should_generate_from_credit_assessment, {"generate": "generate", "escalate": "escalate"}
    )
    graph.add_edge("generate", END)
    graph.add_edge("escalate", END)

    return graph.compile()
