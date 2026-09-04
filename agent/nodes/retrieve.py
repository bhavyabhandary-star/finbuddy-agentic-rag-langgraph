"""Retrieve + Sufficiency-Check nodes — both classical, per kickoff_prompt.md.

The "Add More Context" loop-back edge is implemented as should_continue() in
agent/graph.py, capped at MAX_LOOP_ITERATIONS.
"""
from __future__ import annotations

from agent.state import AgentState
from ingestion.vector_store import VectorStore
from observability.tracing import RunTracer
from tools.rag_tools import retrieve_pdf_chunks

_vector_store = VectorStore()


def retrieve_node(state: AgentState, tracer: RunTracer | None = None) -> AgentState:
    top_k = 5 + 2 * state.get("loop_count", 0)  # widen the search on a loop-back
    result = retrieve_pdf_chunks(state["query"], _vector_store, top_k=top_k)

    if tracer:
        tracer.log_step(
            "retrieve",
            decision=f"retrieved {len(result.chunks)} chunks, top_score={result.top_score:.3f}",
            sufficient=result.sufficient,
            loop_count=state.get("loop_count", 0),
        )

    return {
        **state,
        "retrieved_chunks": result.chunks,
        "top_score": result.top_score,
        "sufficient": result.sufficient,
        "loop_count": state.get("loop_count", 0) + 1,
    }
