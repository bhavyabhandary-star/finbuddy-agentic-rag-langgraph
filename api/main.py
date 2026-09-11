"""API Layer (Layer 1): FastAPI endpoint, request validation, streaming.

Run: uvicorn api.main:app --reload --port 8010
"""
from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agent.graph import build_graph
from agent.nodes.generate import ClaudeProvider, HuggingFaceProvider, LLMProvider, OllamaProvider
from ingestion.setu_feed import load_cached_real_profile
from observability.tracing import trace_run

load_dotenv()

app = FastAPI(title="FinBuddy Agentic RAG (LangGraph)")

# Public read-only demo API, no cookies/session auth to protect -- open CORS
# is the right tradeoff here (a browser-hosted demo UI needs cross-origin
# fetch access, and there's nothing origin-based auth would otherwise gate).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _default_provider() -> LLMProvider:
    """Env-configurable provider selection (LLM_PROVIDER=claude|huggingface|
    ollama, defaults to claude) — kickoff_prompt.md's LLM-agnostic design
    goal, made real: swapping providers for the deployed API needs an env
    var, not a code change. build_graph()'s injectable `provider` param
    (added for testing while Claude billing was blocked) is what makes this
    possible without touching agent/graph.py at all.
    """
    choice = os.environ.get("LLM_PROVIDER", "claude").lower()
    if choice == "huggingface":
        return HuggingFaceProvider()
    if choice == "ollama":
        return OllamaProvider()
    return ClaudeProvider()


_UPI_SIGNAL_KEYS = (
    "avg_monthly_income",
    "income_regularity_score",
    "tx_count_30d",
    "merchant_diversity",
    "balance_dip_frequency",
    "b2b_ratio",
    "avg_transaction_size",
    "tenure_months",
)


class AgentRunRequest(BaseModel):
    query: str
    session_id: str | None = None
    # Present only for a credit-assessment request — the API layer, not the
    # agent, is responsible for collecting these (see agent/state.py's note).
    # risk_trend_delta_features must already be z-scored against the training
    # population — see tools/credit_tools.py's assess_risk_trend docstring.
    credit_signals: dict | None = None
    risk_trend_delta_features: dict | None = None
    # When true, ignores credit_signals and uses the real, most-recently
    # pulled Setu AA sandbox profile instead — the capstone's own "Setu AA
    # Feed" pipeline stage, see ingestion/setu_feed.py. Real sandbox data,
    # not synthesized: consent-based 12-mo UPI pull, normalized to 8 signals.
    use_setu_feed: bool = False


class AgentRunResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    escalate_to_human: bool
    disclaimer: str | None = None
    # Additive decision-trace fields, all optional so existing callers (the
    # verified curl demo commands) are unaffected -- added for a UI that
    # shows the real routing/retrieval decision, not just the final answer.
    route: str | None = None
    route_confidence: float | None = None
    top_score: float | None = None
    sufficient: bool | None = None
    loop_count: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _resolve_credit_signals(request: "AgentRunRequest") -> dict:
    if not request.use_setu_feed:
        return request.credit_signals or {}
    profile = load_cached_real_profile()
    if profile is None:
        return {}  # no real profile pulled yet — assess_credit_profile's own
        # guardrail (tool_error on a bad/incomplete request) covers this, not
        # a fabricated fallback here.
    return {k: profile[k] for k in _UPI_SIGNAL_KEYS if k in profile}


@app.post("/agent/run", response_model=AgentRunResponse)
def run_agent(request: AgentRunRequest) -> AgentRunResponse:
    session_id = request.session_id or str(uuid.uuid4())
    with trace_run(run_id=session_id) as tracer:
        graph = build_graph(tracer=tracer, provider=_default_provider())
        result = graph.invoke(
            {
                "query": request.query,
                "session_id": session_id,
                "credit_signals": _resolve_credit_signals(request),
                "risk_trend_delta_features": request.risk_trend_delta_features,
            }
        )
    return AgentRunResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=result.get("confidence", 0.0),
        escalate_to_human=result.get("escalate_to_human", False),
        disclaimer=result.get("disclaimer"),
        route=result.get("route"),
        route_confidence=result.get("route_confidence"),
        top_score=result.get("top_score"),
        sufficient=result.get("sufficient"),
        loop_count=result.get("loop_count"),
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
            graph = build_graph(tracer=tracer, provider=_default_provider())
            initial_state = {
                "query": request.query,
                "session_id": session_id,
                "credit_signals": _resolve_credit_signals(request),
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
