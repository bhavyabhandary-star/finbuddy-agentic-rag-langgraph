"""API Layer (Layer 1): FastAPI endpoint, request validation, streaming.

Run: uvicorn api.main:app --reload --port 8010
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agent.graph import build_graph
from observability.tracing import trace_run

app = FastAPI(title="FinBuddy Agentic RAG (LangGraph)")


class AgentRunRequest(BaseModel):
    query: str
    session_id: str | None = None
    # Present only for a credit-assessment request — the API layer, not the
    # agent, is responsible for collecting these (see agent/state.py's note).
    credit_signals: dict | None = None
    risk_trend_delta_features: dict | None = None


class AgentRunResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    escalate_to_human: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/agent/run", response_model=AgentRunResponse)
def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    session_id = request.session_id or str(uuid.uuid4())
    with trace_run(run_id=session_id) as tracer:
        graph = build_graph(tracer=tracer)
        result = graph.invoke(
            {
                "query": request.query,
                "session_id": session_id,
                "credit_signals": request.credit_signals or {},
                "risk_trend_delta_features": request.risk_trend_delta_features,
            }
        )
    return AgentRunResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
        escalate_to_human=result.get("escalate_to_human", False),
    )


@app.post("/agent/run/stream")
async def run_agent_stream(request: AgentRunRequest):
    """Streams intermediate step names before the final answer — perceived
    latency matters as much as actual latency (kickoff_prompt.md's Serving
    section: "surface which tool is running, not just final tokens").
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        with trace_run(run_id=session_id) as tracer:
            graph = build_graph(tracer=tracer)
            initial_state = {
                "query": request.query,
                "session_id": session_id,
                "credit_signals": request.credit_signals or {},
                "risk_trend_delta_features": request.risk_trend_delta_features,
            }
            final_state = None
            for step_output in graph.stream(initial_state):
                node_name = next(iter(step_output))
                yield {"event": "step", "data": node_name}
                final_state = step_output[node_name]
            if final_state:
                yield {"event": "final", "data": final_state.get("answer", "")}

    return EventSourceResponse(event_generator())
